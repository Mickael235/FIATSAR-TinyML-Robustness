/*
  ============================================================
  TINY_CONV 12 CLASSES - F1 RUNTIME
  Arduino Nano 33 BLE + TensorFlow Lite Micro

  Objectif :
    - modele TFLite copie en RAM
    - FLIP dynamique d'un poids INT8
    - reconstruction de l'interpreteur TFLM
    - RESTORE dynamique
    - protocole d'inference original conserve

  PC -> Nano
  ----------
  0xA5 + 1960 octets INT8
      inference

  0xB1 + tensor(uint8) + index(uint32 LE) + bit(uint8)
      FLIP

  0xB2
      RESTORE

  0xB3
      STATUS

  Nano -> PC
  ----------
  0x5A + 12 scores + pred + temps(uint32)
      inference

  0xC1 + tensor + index(uint32) + bit + offset(uint32)
       + old(int8) + new(int8)
      FLIP OK

  0xC2 + restored_value(int8)
      RESTORE OK

  0xC3 + active(uint8)
      STATUS

  Erreurs :
  0xED parametre invalide
  0xEE reception incomplete
  0xEF Invoke / reconstruction impossible
  ============================================================
*/

#include <TensorFlowLite.h>
#include <new>
#include <string.h>

#include "model_data.h"
#include "f1_offsets.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ============================================================
// IDENTIFICATION
// ============================================================

const char* MODEL_NAME = "TINY_CONV ORIGINAL - F1 RUNTIME";


// ============================================================
// PROTOCOLE
// ============================================================

constexpr uint8_t CMD_INFERENCE  = 0xA5;
constexpr uint8_t CMD_F1_FLIP    = 0xB1;
constexpr uint8_t CMD_F1_RESTORE = 0xB2;
constexpr uint8_t CMD_F1_STATUS  = 0xB3;

constexpr uint8_t RESP_INFERENCE  = 0x5A;
constexpr uint8_t RESP_F1_FLIP    = 0xC1;
constexpr uint8_t RESP_F1_RESTORE = 0xC2;
constexpr uint8_t RESP_F1_STATUS  = 0xC3;

constexpr uint8_t ERROR_PARAM  = 0xED;
constexpr uint8_t ERROR_RX     = 0xEE;
constexpr uint8_t ERROR_INVOKE = 0xEF;

constexpr int INPUT_SIZE  = 1960;
constexpr int OUTPUT_SIZE = 12;

constexpr uint32_t RX_TIMEOUT_MS = 10000;


// ============================================================
// MEMOIRE
// ============================================================

constexpr int kTensorArenaSize = 16 * 1024;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Copie du .tflite modifiable en RAM.
alignas(16) static uint8_t model_ram[F1_MODEL_SIZE];

// Stockage statique de l'objet MicroInterpreter.
alignas(tflite::MicroInterpreter)
static uint8_t interpreter_storage[
    sizeof(tflite::MicroInterpreter)
];


// ============================================================
// TFLM
// ============================================================

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::MicroMutableOpResolver<4> resolver;

bool interpreter_constructed = false;

}  // namespace


// ============================================================
// ETAT FAUTE F1
// ============================================================

struct F1ActiveFault {
  bool active;
  uint8_t tensor_index;
  uint32_t weight_index;
  uint8_t bit;
  uint32_t absolute_offset;
  uint8_t original_byte;
  uint8_t faulted_byte;
};

static F1ActiveFault active_fault = {
    false, 0, 0, 0, 0, 0, 0
};


// ============================================================
// OUTILS
// ============================================================

void fatalError(const char* message) {
  Serial.print("ERREUR FATALE : ");
  Serial.println(message);

  while (true) {
    delay(1000);
  }
}


bool readExactBytes(
    uint8_t* destination,
    size_t length,
    uint32_t timeout_ms = RX_TIMEOUT_MS
) {
  size_t received = 0;
  uint32_t last_receive = millis();

  while (received < length) {
    int availableBytes = Serial.available();

    if (availableBytes > 0) {
      size_t remaining = length - received;

      size_t toRead =
          (availableBytes < (int)remaining)
          ? availableBytes
          : remaining;

      int n = Serial.readBytes(
          reinterpret_cast<char*>(
              destination + received
          ),
          toRead
      );

      if (n > 0) {
        received += n;
        last_receive = millis();
      }
    }

    if (millis() - last_receive > timeout_ms) {
      return false;
    }
  }

  return true;
}


uint32_t readU32LE(const uint8_t* p) {
  return
      static_cast<uint32_t>(p[0])
      | (static_cast<uint32_t>(p[1]) << 8)
      | (static_cast<uint32_t>(p[2]) << 16)
      | (static_cast<uint32_t>(p[3]) << 24);
}


void writeU32LE(uint32_t value) {
  uint8_t b[4] = {
      static_cast<uint8_t>(value & 0xFF),
      static_cast<uint8_t>((value >> 8) & 0xFF),
      static_cast<uint8_t>((value >> 16) & 0xFF),
      static_cast<uint8_t>((value >> 24) & 0xFF)
  };

  Serial.write(b, 4);
}


uint8_t argmax12(const int8_t* scores) {
  uint8_t prediction = 0;
  int8_t best_score = scores[0];

  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      prediction = static_cast<uint8_t>(i);
    }
  }

  return prediction;
}


bool findWeightTensor(
    uint8_t tensor_index,
    size_t* data_offset,
    size_t* byte_count
) {
  for (size_t i = 0; i < F1_WEIGHT_TENSOR_COUNT; i++) {
    if (
        F1_WEIGHT_TENSORS[i].tensor_index
        == tensor_index
    ) {
      *data_offset =
          F1_WEIGHT_TENSORS[i].data_offset;

      *byte_count =
          F1_WEIGHT_TENSORS[i].byte_count;

      return true;
    }
  }

  return false;
}


// ============================================================
// RECONSTRUCTION INTERPRETEUR
// ============================================================

bool rebuildInterpreter() {
  if (interpreter_constructed && interpreter != nullptr) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
    input = nullptr;
    output = nullptr;
    interpreter_constructed = false;
  }

  memset(
      tensor_arena,
      0,
      sizeof(tensor_arena)
  );

  model = tflite::GetModel(model_ram);

  if (model == nullptr) {
    return false;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return false;
  }

  interpreter =
      new (interpreter_storage)
      tflite::MicroInterpreter(
          model,
          resolver,
          tensor_arena,
          kTensorArenaSize,
          nullptr,
          nullptr
      );

  interpreter_constructed = true;

  if (
      interpreter->AllocateTensors()
      != kTfLiteOk
  ) {
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    return false;
  }

  if (
      input->type != kTfLiteInt8
      || input->dims->size != 2
      || input->dims->data[0] != 1
      || input->dims->data[1] != INPUT_SIZE
  ) {
    return false;
  }

  if (
      output->type != kTfLiteInt8
      || output->dims->size != 2
      || output->dims->data[0] != 1
      || output->dims->data[1] != OUTPUT_SIZE
  ) {
    return false;
  }

  return true;
}


// ============================================================
// F1 FLIP
// ============================================================

bool applyFlip(
    uint8_t tensor_index,
    uint32_t weight_index,
    uint8_t bit
) {
  if (active_fault.active) {
    return false;
  }

  if (bit > 7) {
    return false;
  }

  size_t data_offset = 0;
  size_t byte_count = 0;

  if (
      !findWeightTensor(
          tensor_index,
          &data_offset,
          &byte_count
      )
  ) {
    return false;
  }

  if (weight_index >= byte_count) {
    return false;
  }

  uint32_t absolute_offset =
      static_cast<uint32_t>(
          data_offset + weight_index
      );

  if (absolute_offset >= F1_MODEL_SIZE) {
    return false;
  }

  uint8_t old_byte =
      model_ram[absolute_offset];

  uint8_t new_byte =
      old_byte
      ^ static_cast<uint8_t>(1u << bit);

  model_ram[absolute_offset] =
      new_byte;

  active_fault.active = true;
  active_fault.tensor_index = tensor_index;
  active_fault.weight_index = weight_index;
  active_fault.bit = bit;
  active_fault.absolute_offset = absolute_offset;
  active_fault.original_byte = old_byte;
  active_fault.faulted_byte = new_byte;

  if (!rebuildInterpreter()) {
    // rollback
    model_ram[absolute_offset] =
        old_byte;

    active_fault.active = false;

    rebuildInterpreter();

    return false;
  }

  return true;
}


// ============================================================
// F1 RESTORE
// ============================================================

bool restoreFault(
    uint8_t* restored_value
) {
  if (!active_fault.active) {
    *restored_value = 0;
    return true;
  }

  const uint32_t absolute_offset =
      active_fault.absolute_offset;

  const uint8_t original =
      active_fault.original_byte;

  model_ram[absolute_offset] =
      original;

  *restored_value =
      original;

  active_fault.active = false;

  if (!rebuildInterpreter()) {
    return false;
  }

  return true;
}


// ============================================================
// REPONSES
// ============================================================

void sendInferenceResponse(
    uint32_t inference_us
) {
  uint8_t prediction =
      argmax12(
          output->data.int8
      );

  Serial.write(
      RESP_INFERENCE
  );

  Serial.write(
      reinterpret_cast<uint8_t*>(
          output->data.int8
      ),
      OUTPUT_SIZE
  );

  Serial.write(
      prediction
  );

  writeU32LE(
      inference_us
  );

  Serial.flush();
}


void sendFlipResponse() {
  Serial.write(
      RESP_F1_FLIP
  );

  Serial.write(
      active_fault.tensor_index
  );

  writeU32LE(
      active_fault.weight_index
  );

  Serial.write(
      active_fault.bit
  );

  writeU32LE(
      active_fault.absolute_offset
  );

  Serial.write(
      active_fault.original_byte
  );

  Serial.write(
      active_fault.faulted_byte
  );

  Serial.flush();
}


void sendRestoreResponse(
    uint8_t restored_value
) {
  Serial.write(
      RESP_F1_RESTORE
  );

  Serial.write(
      restored_value
  );

  Serial.flush();
}


void sendStatusResponse() {
  Serial.write(
      RESP_F1_STATUS
  );

  Serial.write(
      active_fault.active ? 1 : 0
  );

  Serial.flush();
}


// ============================================================
// INFOS DEMARRAGE
// ============================================================

void printModelInfo() {
  Serial.println();
  Serial.println("========================================");
  Serial.println(" TINY_CONV - F1 RUNTIME");
  Serial.println("========================================");

  Serial.print("Modele : ");
  Serial.println(MODEL_NAME);

  Serial.print("Taille modele Flash : ");
  Serial.print(g_model_len);
  Serial.println(" octets");

  Serial.print("Taille copie RAM    : ");
  Serial.print(F1_MODEL_SIZE);
  Serial.println(" octets");

  Serial.print("Tensor arena utilisee : ");
  Serial.print(
      static_cast<int>(
          interpreter->arena_used_bytes()
      )
  );
  Serial.println(" octets");

  Serial.print("Tensor arena reservee : ");
  Serial.print(kTensorArenaSize);
  Serial.println(" octets");

  Serial.println();
  Serial.println("Commandes :");
  Serial.println("0xA5 = inference");
  Serial.println("0xB1 = F1 FLIP");
  Serial.println("0xB2 = F1 RESTORE");
  Serial.println("0xB3 = F1 STATUS");
  Serial.println("========================================");
}


// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);

  unsigned long start = millis();

  while (
      !Serial
      && (millis() - start < 10000)
  ) {
    delay(10);
  }

  Serial.setTimeout(
      RX_TIMEOUT_MS
  );

  if (
      static_cast<size_t>(g_model_len)
      != F1_MODEL_SIZE
  ) {
    fatalError(
        "MODEL_SIZE_MISMATCH"
    );
  }

  memcpy(
      model_ram,
      g_model,
      F1_MODEL_SIZE
  );

  if (
      resolver.AddConv2D()
      != kTfLiteOk
  ) {
    fatalError("CONV2D");
  }

  if (
      resolver.AddFullyConnected()
      != kTfLiteOk
  ) {
    fatalError("FULLY_CONNECTED");
  }

  if (
      resolver.AddSoftmax()
      != kTfLiteOk
  ) {
    fatalError("SOFTMAX");
  }

  if (
      resolver.AddReshape()
      != kTfLiteOk
  ) {
    fatalError("RESHAPE");
  }

  if (!rebuildInterpreter()) {
    fatalError(
        "INITIAL_REBUILD"
    );
  }

  printModelInfo();

  Serial.println("READY");
  Serial.flush();
}


// ============================================================
// LOOP
// ============================================================

void loop() {
  if (
      Serial.available() <= 0
  ) {
    return;
  }

  const uint8_t command =
      Serial.read();


  // ----------------------------------------------------------
  // INFERENCE
  // ----------------------------------------------------------

  if (
      command == CMD_INFERENCE
  ) {
    bool received_ok =
        readExactBytes(
            reinterpret_cast<uint8_t*>(
                input->data.int8
            ),
            INPUT_SIZE,
            RX_TIMEOUT_MS
        );

    if (!received_ok) {
      Serial.write(ERROR_RX);
      Serial.flush();
      return;
    }

    uint32_t start_us =
        micros();

    TfLiteStatus status =
        interpreter->Invoke();

    uint32_t inference_us =
        micros() - start_us;

    if (
        status != kTfLiteOk
    ) {
      Serial.write(
          ERROR_INVOKE
      );

      Serial.flush();

      return;
    }

    sendInferenceResponse(
        inference_us
    );

    return;
  }


  // ----------------------------------------------------------
  // F1 FLIP
  // Payload :
  // tensor uint8
  // index uint32 little endian
  // bit uint8
  // ----------------------------------------------------------

  if (
      command == CMD_F1_FLIP
  ) {
    uint8_t payload[6];

    if (
        !readExactBytes(
            payload,
            sizeof(payload),
            RX_TIMEOUT_MS
        )
    ) {
      Serial.write(ERROR_RX);
      Serial.flush();
      return;
    }

    uint8_t tensor_index =
        payload[0];

    uint32_t weight_index =
        readU32LE(
            &payload[1]
        );

    uint8_t bit =
        payload[5];

    if (
        !applyFlip(
            tensor_index,
            weight_index,
            bit
        )
    ) {
      Serial.write(
          ERROR_PARAM
      );

      Serial.flush();

      return;
    }

    sendFlipResponse();

    return;
  }


  // ----------------------------------------------------------
  // F1 RESTORE
  // ----------------------------------------------------------

  if (
      command == CMD_F1_RESTORE
  ) {
    uint8_t restored_value = 0;

    if (
        !restoreFault(
            &restored_value
        )
    ) {
      Serial.write(
          ERROR_INVOKE
      );

      Serial.flush();

      return;
    }

    sendRestoreResponse(
        restored_value
    );

    return;
  }


  // ----------------------------------------------------------
  // F1 STATUS
  // ----------------------------------------------------------

  if (
      command == CMD_F1_STATUS
  ) {
    sendStatusResponse();
    return;
  }
}
