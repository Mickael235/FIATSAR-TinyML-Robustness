/*
  ============================================================
  TINY_CONV 12 CLASSES - F3 FULLY CONNECTED
  Arduino Nano 33 BLE + TFLite Micro ancienne API

  F3 simule une sortie prematuree de la boucle d'accumulation
  de la FullyConnected :
      seules les k premieres contributions sont conservees ;
      les activations [k..3999] sont placees au zero-point.

  MODELE PRINCIPAL :
      f3_model_original_with_fc_input.tflite
      output(0) = Softmax [1,12]
      output(1) = entree FC / tensor 9 [1,4000]

  SOUS-MODELE :
      f2_tail_fc_softmax.tflite
      entree [1,4000] -> FullyConnected -> Softmax

  COMMANDES PC -> Nano :

  Nominal :
      0xA5 + 1960 octets INT8

  F3 FullyConnected :
      0xA8
      + k uint16 little-endian
      + 1960 octets INT8

  REPONSES Nano -> PC :

  Nominal :
      0x5A + 12 scores + prediction + temps uint32

  F3 :
      0x5C + 12 scores + prediction + temps uint32 + k uint16

  La faute est transitoire : l'entree FC est recopiee a neuf
  a chaque commande avant d'etre tronquee.
  ============================================================
*/

#include <TensorFlowLite.h>

#include "model_data.h"
#include "fc_tail_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr uint8_t CMD_NOMINAL = 0xA5;
constexpr uint8_t CMD_F3_FC   = 0xA8;

constexpr uint8_t RESP_NOMINAL = 0x5A;
constexpr uint8_t RESP_F3      = 0x5C;

constexpr uint8_t ERROR_PARAM  = 0xED;
constexpr uint8_t ERROR_RX     = 0xEE;
constexpr uint8_t ERROR_INVOKE = 0xEF;

constexpr int INPUT_SIZE = 1960;
constexpr int OUTPUT_SIZE = 12;
constexpr int FC_INPUT_SIZE = 4000;

constexpr int kMainArenaSize = 24 * 1024;
constexpr int kTailArenaSize = 24 * 1024;

constexpr uint32_t RX_TIMEOUT_MS = 10000;


namespace {

const tflite::Model* main_model = nullptr;
const tflite::Model* tail_model = nullptr;

tflite::MicroInterpreter* main_interpreter = nullptr;
tflite::MicroInterpreter* tail_interpreter = nullptr;

TfLiteTensor* main_input = nullptr;
TfLiteTensor* main_output = nullptr;
TfLiteTensor* main_fc_input = nullptr;

TfLiteTensor* tail_input = nullptr;
TfLiteTensor* tail_output = nullptr;

alignas(16) uint8_t main_arena[kMainArenaSize];
alignas(16) uint8_t tail_arena[kTailArenaSize];

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
          reinterpret_cast<char*>(destination + received),
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


void sendNominal(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us
) {
  Serial.write(RESP_NOMINAL);
  Serial.write(
      reinterpret_cast<const uint8_t*>(scores),
      OUTPUT_SIZE
  );
  Serial.write(prediction);
  Serial.write(
      reinterpret_cast<uint8_t*>(&inference_us),
      sizeof(inference_us)
  );
  Serial.flush();
}


void sendF3(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us,
    uint16_t k
) {
  Serial.write(RESP_F3);
  Serial.write(
      reinterpret_cast<const uint8_t*>(scores),
      OUTPUT_SIZE
  );
  Serial.write(prediction);
  Serial.write(
      reinterpret_cast<uint8_t*>(&inference_us),
      sizeof(inference_us)
  );
  Serial.write(
      reinterpret_cast<uint8_t*>(&k),
      sizeof(k)
  );
  Serial.flush();
}


void printInfo() {
  Serial.println();
  Serial.println("========================================");
  Serial.println(" TINY_CONV - F3 FULLY CONNECTED");
  Serial.println("========================================");

  Serial.print("Main model : ");
  Serial.print(g_model_len);
  Serial.println(" octets");

  Serial.print("FC tail : ");
  Serial.print(g_fc_tail_model_len);
  Serial.println(" octets");

  Serial.println("Main output(0) : Softmax [1,12]");
  Serial.println("Main output(1) : FC input [1,4000]");

  Serial.print("FC zero-point : ");
  Serial.println(tail_input->params.zero_point);

  Serial.println();

  Serial.print("Main arena utilisee : ");
  Serial.println(
      static_cast<int>(
          main_interpreter->arena_used_bytes()
      )
  );

  Serial.print("Main arena reservee : ");
  Serial.println(kMainArenaSize);

  Serial.print("Tail arena utilisee : ");
  Serial.println(
      static_cast<int>(
          tail_interpreter->arena_used_bytes()
      )
  );

  Serial.print("Tail arena reservee : ");
  Serial.println(kTailArenaSize);

  Serial.println();
  Serial.println("0xA5 = nominal");
  Serial.println("0xA8 = F3 FC, parametre k uint16");
  Serial.println("========================================");
}


void setup() {
  Serial.begin(115200);

  unsigned long start = millis();
  while (!Serial && (millis() - start < 10000)) {
    delay(10);
  }

  Serial.setTimeout(RX_TIMEOUT_MS);


  // ----------------------------------------------------------
  // MODELE PRINCIPAL
  // ----------------------------------------------------------

  main_model = tflite::GetModel(g_model);

  if (main_model == nullptr) {
    fatalError("MAIN_MODEL_NULL");
  }

  if (main_model->version() != TFLITE_SCHEMA_VERSION) {
    fatalError("MAIN_BAD_SCHEMA");
  }

  static tflite::MicroMutableOpResolver<4> main_resolver;

  if (main_resolver.AddConv2D() != kTfLiteOk) {
    fatalError("CONV2D");
  }
  if (main_resolver.AddFullyConnected() != kTfLiteOk) {
    fatalError("FULLY_CONNECTED");
  }
  if (main_resolver.AddSoftmax() != kTfLiteOk) {
    fatalError("SOFTMAX");
  }
  if (main_resolver.AddReshape() != kTfLiteOk) {
    fatalError("RESHAPE");
  }

  static tflite::MicroInterpreter static_main_interpreter(
      main_model,
      main_resolver,
      main_arena,
      kMainArenaSize,
      nullptr,
      nullptr
  );

  main_interpreter = &static_main_interpreter;

  if (main_interpreter->AllocateTensors() != kTfLiteOk) {
    fatalError("MAIN_ALLOCATE");
  }

  main_input = main_interpreter->input(0);
  main_output = main_interpreter->output(0);
  main_fc_input = main_interpreter->output(1);

  if (
      main_input == nullptr
      || main_output == nullptr
      || main_fc_input == nullptr
  ) {
    fatalError("MAIN_TENSOR_NULL");
  }

  if (
      main_fc_input->type != kTfLiteInt8
      || main_fc_input->dims->size != 2
      || main_fc_input->dims->data[0] != 1
      || main_fc_input->dims->data[1] != FC_INPUT_SIZE
  ) {
    fatalError("BAD_FC_INPUT_OUTPUT");
  }


  // ----------------------------------------------------------
  // TAIL FC + SOFTMAX
  // ----------------------------------------------------------

  tail_model = tflite::GetModel(g_fc_tail_model);

  if (tail_model == nullptr) {
    fatalError("TAIL_MODEL_NULL");
  }

  if (tail_model->version() != TFLITE_SCHEMA_VERSION) {
    fatalError("TAIL_BAD_SCHEMA");
  }

  static tflite::MicroMutableOpResolver<2> tail_resolver;

  if (tail_resolver.AddFullyConnected() != kTfLiteOk) {
    fatalError("TAIL_FC");
  }
  if (tail_resolver.AddSoftmax() != kTfLiteOk) {
    fatalError("TAIL_SOFTMAX");
  }

  static tflite::MicroInterpreter static_tail_interpreter(
      tail_model,
      tail_resolver,
      tail_arena,
      kTailArenaSize,
      nullptr,
      nullptr
  );

  tail_interpreter = &static_tail_interpreter;

  if (tail_interpreter->AllocateTensors() != kTfLiteOk) {
    fatalError("TAIL_ALLOCATE");
  }

  tail_input = tail_interpreter->input(0);
  tail_output = tail_interpreter->output(0);

  if (
      tail_input == nullptr
      || tail_output == nullptr
  ) {
    fatalError("TAIL_TENSOR_NULL");
  }

  if (
      tail_input->type != kTfLiteInt8
      || tail_input->dims->size != 2
      || tail_input->dims->data[0] != 1
      || tail_input->dims->data[1] != FC_INPUT_SIZE
  ) {
    fatalError("TAIL_BAD_INPUT");
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
      && command != CMD_F3_FC
  ) {
    return;
  }

  uint16_t k = FC_INPUT_SIZE;

  if (command == CMD_F3_FC) {
    uint8_t raw_k[2];

    if (!readExactBytes(raw_k, 2, RX_TIMEOUT_MS)) {
      Serial.write(ERROR_RX);
      Serial.flush();
      drainSerialUntilQuiet();
      return;
    }

    k =
        static_cast<uint16_t>(raw_k[0])
        | (
            static_cast<uint16_t>(raw_k[1])
            << 8
        );

    if (k > FC_INPUT_SIZE) {
      Serial.write(ERROR_PARAM);
      Serial.flush();
      return;
    }
  }


  if (
      !readExactBytes(
          reinterpret_cast<uint8_t*>(
              main_input->data.int8
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

  if (main_interpreter->Invoke() != kTfLiteOk) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }


  if (command == CMD_NOMINAL) {
    uint32_t inference_us =
        micros() - start_us;

    uint8_t prediction =
        argmax12(
            main_output->data.int8
        );

    sendNominal(
        main_output->data.int8,
        prediction,
        inference_us
    );

    return;
  }


  // ----------------------------------------------------------
  // F3 : COPIE FRAICHE DE L'ENTREE FC
  // ----------------------------------------------------------

  for (int i = 0; i < FC_INPUT_SIZE; i++) {
    tail_input->data.int8[i] =
        main_fc_input->data.int8[i];
  }


  // ----------------------------------------------------------
  // SORTIE PREMATUREE APRES k CONTRIBUTIONS
  // ----------------------------------------------------------

  const int8_t zero_point =
      static_cast<int8_t>(
          tail_input->params.zero_point
      );

  for (int i = k; i < FC_INPUT_SIZE; i++) {
    tail_input->data.int8[i] =
        zero_point;
  }


  if (tail_interpreter->Invoke() != kTfLiteOk) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }


  uint32_t inference_us =
      micros() - start_us;

  uint8_t prediction =
      argmax12(
          tail_output->data.int8
      );


  sendF3(
      tail_output->data.int8,
      prediction,
      inference_us,
      k
  );
}
