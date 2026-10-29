/*
  ============================================================
  TINY_CONV 12 CLASSES - F3 CONV2D k=79/80
  Arduino Nano 33 BLE + ancienne API TFLite Micro

  Reproduction fonctionnelle du point de coupure F3 Conv2D :
      k = 79 / 80
      la derniere contribution MAC de chaque filtre est annulee.

  Deux modeles sont embarques :
      - nominal : model_data.h
      - F3 Conv k79 : conv_fault_model_data.h

  La commande choisit le modele pour UNE inference.
  L'inference suivante peut repartir sur le modele nominal.

  PC -> Nano :
      0xA5 + 1960 octets INT8 : nominal
      0xA9 + 1960 octets INT8 : F3 Conv k79

  Nano -> PC :
      0x5A + 12 scores + prediction + temps uint32
      0x5D + 12 scores + prediction + temps uint32 + k uint8
  ============================================================
*/

#include <TensorFlowLite.h>

#include "model_data.h"
#include "conv_fault_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr uint8_t CMD_NOMINAL = 0xA5;
constexpr uint8_t CMD_F3_CONV = 0xA9;

constexpr uint8_t RESP_NOMINAL = 0x5A;
constexpr uint8_t RESP_F3_CONV = 0x5D;

constexpr uint8_t ERROR_RX = 0xEE;
constexpr uint8_t ERROR_INVOKE = 0xEF;

constexpr int INPUT_SIZE = 1960;
constexpr int OUTPUT_SIZE = 12;
constexpr uint8_t F3_K = 79;

constexpr uint32_t RX_TIMEOUT_MS = 10000;

constexpr int kNominalArenaSize = 16 * 1024;
constexpr int kFaultArenaSize   = 16 * 1024;


namespace {

const tflite::Model* nominal_model = nullptr;
const tflite::Model* fault_model = nullptr;

tflite::MicroInterpreter* nominal_interpreter = nullptr;
tflite::MicroInterpreter* fault_interpreter = nullptr;

TfLiteTensor* nominal_input = nullptr;
TfLiteTensor* nominal_output = nullptr;

TfLiteTensor* fault_input = nullptr;
TfLiteTensor* fault_output = nullptr;

alignas(16) uint8_t nominal_arena[kNominalArenaSize];
alignas(16) uint8_t fault_arena[kFaultArenaSize];

}  // namespace


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


void drainSerialUntilQuiet(
    uint32_t quiet_ms = 50,
    uint32_t max_ms = 1000
) {
  uint32_t start = millis();
  uint32_t last_byte = millis();

  while (millis() - start < max_ms) {
    bool received = false;

    while (Serial.available() > 0) {
      Serial.read();
      received = true;
      last_byte = millis();
    }

    if (!received && (millis() - last_byte >= quiet_ms)) {
      break;
    }

    delay(1);
  }
}


uint8_t argmax12(const int8_t* scores) {
  uint8_t prediction = 0;
  int8_t best = scores[0];

  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (scores[i] > best) {
      best = scores[i];
      prediction = static_cast<uint8_t>(i);
    }
  }

  return prediction;
}


void sendResponse(
    uint8_t header,
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us,
    bool append_k
) {
  Serial.write(header);

  Serial.write(
      reinterpret_cast<const uint8_t*>(scores),
      OUTPUT_SIZE
  );

  Serial.write(prediction);

  Serial.write(
      reinterpret_cast<uint8_t*>(&inference_us),
      sizeof(inference_us)
  );

  if (append_k) {
    Serial.write(F3_K);
  }

  Serial.flush();
}


void printInfo() {
  Serial.println();
  Serial.println("========================================");
  Serial.println(" TINY_CONV - F3 CONV2D k=79/80");
  Serial.println("========================================");

  Serial.print("Modele nominal : ");
  Serial.print(g_model_len);
  Serial.println(" octets");

  Serial.print("Modele F3 Conv : ");
  Serial.print(g_conv_fault_model_len);
  Serial.println(" octets");

  Serial.print("Nominal arena utilisee : ");
  Serial.println(
      static_cast<int>(
          nominal_interpreter->arena_used_bytes()
      )
  );

  Serial.print("Fault arena utilisee   : ");
  Serial.println(
      static_cast<int>(
          fault_interpreter->arena_used_bytes()
      )
  );

  Serial.println();
  Serial.println("0xA5 = nominal");
  Serial.println("0xA9 = F3 Conv k79/80");
  Serial.println("========================================");
}


void setup() {
  Serial.begin(115200);

  unsigned long start = millis();
  while (!Serial && (millis() - start < 10000)) {
    delay(10);
  }

  Serial.setTimeout(RX_TIMEOUT_MS);

  nominal_model = tflite::GetModel(g_model);
  fault_model = tflite::GetModel(g_conv_fault_model);

  if (
      nominal_model == nullptr
      || fault_model == nullptr
  ) {
    fatalError("MODEL_NULL");
  }

  if (
      nominal_model->version() != TFLITE_SCHEMA_VERSION
      || fault_model->version() != TFLITE_SCHEMA_VERSION
  ) {
    fatalError("BAD_SCHEMA");
  }

  static tflite::MicroMutableOpResolver<4> resolver;

  if (resolver.AddConv2D() != kTfLiteOk) {
    fatalError("CONV2D");
  }
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    fatalError("FULLY_CONNECTED");
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    fatalError("SOFTMAX");
  }
  if (resolver.AddReshape() != kTfLiteOk) {
    fatalError("RESHAPE");
  }

  static tflite::MicroInterpreter static_nominal_interpreter(
      nominal_model,
      resolver,
      nominal_arena,
      kNominalArenaSize,
      nullptr,
      nullptr
  );

  static tflite::MicroInterpreter static_fault_interpreter(
      fault_model,
      resolver,
      fault_arena,
      kFaultArenaSize,
      nullptr,
      nullptr
  );

  nominal_interpreter = &static_nominal_interpreter;
  fault_interpreter = &static_fault_interpreter;

  if (nominal_interpreter->AllocateTensors() != kTfLiteOk) {
    fatalError("NOMINAL_ALLOCATE");
  }

  if (fault_interpreter->AllocateTensors() != kTfLiteOk) {
    fatalError("FAULT_ALLOCATE");
  }

  nominal_input = nominal_interpreter->input(0);
  nominal_output = nominal_interpreter->output(0);

  fault_input = fault_interpreter->input(0);
  fault_output = fault_interpreter->output(0);

  if (
      nominal_input == nullptr
      || nominal_output == nullptr
      || fault_input == nullptr
      || fault_output == nullptr
  ) {
    fatalError("TENSOR_NULL");
  }

  printInfo();

  Serial.println("READY");
  Serial.flush();
}


void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  uint8_t command = Serial.read();

  if (
      command != CMD_NOMINAL
      && command != CMD_F3_CONV
  ) {
    return;
  }

  tflite::MicroInterpreter* selected_interpreter = nullptr;
  TfLiteTensor* selected_input = nullptr;
  TfLiteTensor* selected_output = nullptr;
  uint8_t response_header = 0;
  bool append_k = false;

  if (command == CMD_NOMINAL) {
    selected_interpreter = nominal_interpreter;
    selected_input = nominal_input;
    selected_output = nominal_output;
    response_header = RESP_NOMINAL;
    append_k = false;
  }
  else {
    selected_interpreter = fault_interpreter;
    selected_input = fault_input;
    selected_output = fault_output;
    response_header = RESP_F3_CONV;
    append_k = true;
  }

  if (
      !readExactBytes(
          reinterpret_cast<uint8_t*>(
              selected_input->data.int8
          ),
          INPUT_SIZE,
          RX_TIMEOUT_MS
      )
  ) {
    Serial.write(ERROR_RX);
    Serial.flush();
    drainSerialUntilQuiet();
    return;
  }

  uint32_t start_us = micros();

  if (selected_interpreter->Invoke() != kTfLiteOk) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }

  uint32_t inference_us = micros() - start_us;

  uint8_t prediction =
      argmax12(
          selected_output->data.int8
      );

  sendResponse(
      response_header,
      selected_output->data.int8,
      prediction,
      inference_us,
      append_k
  );
}
