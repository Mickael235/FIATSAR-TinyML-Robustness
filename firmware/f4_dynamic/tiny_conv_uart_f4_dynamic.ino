/*
  ============================================================
  TINY_CONV 12 CLASSES - F4 DYNAMIQUE
  Arduino Nano 33 BLE + TFLite Micro

  F4-A : argmax/sortie tronquee apres k classes
         commande 0xAA + k + 1960 octets INT8

  F4-B : score force au maximum INT8 (127)
         commande 0xAB + class_id + 1960 octets INT8

  Nominal :
         0xA5 + 1960 octets INT8

  Reponses :
      0x5A + 12 scores + pred + temps uint32
      0x5E + 12 scores + pred + temps uint32 + k
      0x5F + 12 scores + pred + temps uint32
           + class_id + old_score + new_score

  La faute est transitoire :
  le tensor de sortie TFLM n'est pas modifie durablement.
  ============================================================
*/

#include <TensorFlowLite.h>

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr uint8_t CMD_NOMINAL      = 0xA5;
constexpr uint8_t CMD_F4_TRUNCATE  = 0xAA;
constexpr uint8_t CMD_F4_FORCE     = 0xAB;

constexpr uint8_t RESP_NOMINAL     = 0x5A;
constexpr uint8_t RESP_F4_TRUNCATE = 0x5E;
constexpr uint8_t RESP_F4_FORCE    = 0x5F;

constexpr uint8_t ERROR_PARAM  = 0xED;
constexpr uint8_t ERROR_RX     = 0xEE;
constexpr uint8_t ERROR_INVOKE = 0xEF;

constexpr int INPUT_SIZE = 1960;
constexpr int OUTPUT_SIZE = 12;

constexpr uint32_t RX_TIMEOUT_MS = 10000;
constexpr int kTensorArenaSize = 16 * 1024;


namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int8_t fault_scores[OUTPUT_SIZE];

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


uint8_t argmaxN(
    const int8_t* scores,
    uint8_t count
) {
  uint8_t prediction = 0;
  int8_t best = scores[0];

  for (uint8_t i = 1; i < count; i++) {
    if (scores[i] > best) {
      best = scores[i];
      prediction = i;
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


void sendTruncate(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us,
    uint8_t k
) {
  Serial.write(RESP_F4_TRUNCATE);

  // Les scores restent ceux du Softmax nominal.
  // La faute porte sur l'interruption de l'argmax.
  Serial.write(
      reinterpret_cast<const uint8_t*>(scores),
      OUTPUT_SIZE
  );

  Serial.write(prediction);

  Serial.write(
      reinterpret_cast<uint8_t*>(&inference_us),
      sizeof(inference_us)
  );

  Serial.write(k);
  Serial.flush();
}


void sendForce(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us,
    uint8_t class_id,
    int8_t old_score,
    int8_t new_score
) {
  Serial.write(RESP_F4_FORCE);

  Serial.write(
      reinterpret_cast<const uint8_t*>(scores),
      OUTPUT_SIZE
  );

  Serial.write(prediction);

  Serial.write(
      reinterpret_cast<uint8_t*>(&inference_us),
      sizeof(inference_us)
  );

  Serial.write(class_id);
  Serial.write(
      reinterpret_cast<uint8_t*>(&old_score),
      1
  );
  Serial.write(
      reinterpret_cast<uint8_t*>(&new_score),
      1
  );

  Serial.flush();
}


void printInfo() {
  Serial.println();
  Serial.println("========================================");
  Serial.println(" TINY_CONV - F4 DYNAMIQUE");
  Serial.println("========================================");

  Serial.print("Modele : ");
  Serial.print(g_model_len);
  Serial.println(" octets");

  Serial.print("Arena utilisee : ");
  Serial.println(
      static_cast<int>(
          interpreter->arena_used_bytes()
      )
  );

  Serial.print("Arena reservee : ");
  Serial.println(kTensorArenaSize);

  Serial.println();
  Serial.println("0xA5 = nominal");
  Serial.println("0xAA = F4 truncate argmax");
  Serial.println("0xAB = F4 force score");
  Serial.println("========================================");
}


void setup() {
  Serial.begin(115200);

  unsigned long start = millis();

  while (!Serial && (millis() - start < 10000)) {
    delay(10);
  }

  Serial.setTimeout(RX_TIMEOUT_MS);

  model = tflite::GetModel(g_model);

  if (model == nullptr) {
    fatalError("MODEL_NULL");
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
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

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      nullptr,
      nullptr
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatalError("ALLOCATE");
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
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
      && command != CMD_F4_TRUNCATE
      && command != CMD_F4_FORCE
  ) {
    return;
  }

  uint8_t parameter = 0;

  if (
      command == CMD_F4_TRUNCATE
      || command == CMD_F4_FORCE
  ) {
    if (!readExactBytes(&parameter, 1, RX_TIMEOUT_MS)) {
      Serial.write(ERROR_RX);
      Serial.flush();
      drainSerialUntilQuiet();
      return;
    }
  }

  if (
      command == CMD_F4_TRUNCATE
      && (parameter < 1 || parameter > OUTPUT_SIZE)
  ) {
    Serial.write(ERROR_PARAM);
    Serial.flush();
    return;
  }

  if (
      command == CMD_F4_FORCE
      && parameter >= OUTPUT_SIZE
  ) {
    Serial.write(ERROR_PARAM);
    Serial.flush();
    return;
  }

  if (
      !readExactBytes(
          reinterpret_cast<uint8_t*>(
              input->data.int8
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

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }

  // ----------------------------------------------------------
  // NOMINAL
  // ----------------------------------------------------------

  if (command == CMD_NOMINAL) {
    uint32_t inference_us =
        micros() - start_us;

    uint8_t prediction =
        argmaxN(
            output->data.int8,
            OUTPUT_SIZE
        );

    sendNominal(
        output->data.int8,
        prediction,
        inference_us
    );

    return;
  }

  // ----------------------------------------------------------
  // F4-A : ARGMAX TRONQUE
  // ----------------------------------------------------------

  if (command == CMD_F4_TRUNCATE) {
    const uint8_t k = parameter;

    uint8_t prediction =
        argmaxN(
            output->data.int8,
            k
        );

    uint32_t inference_us =
        micros() - start_us;

    sendTruncate(
        output->data.int8,
        prediction,
        inference_us,
        k
    );

    return;
  }

  // ----------------------------------------------------------
  // F4-B : SCORE FORCE A 127
  // ----------------------------------------------------------

  const uint8_t class_id = parameter;

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    fault_scores[i] =
        output->data.int8[i];
  }

  const int8_t old_score =
      fault_scores[class_id];

  const int8_t new_score = 127;

  fault_scores[class_id] =
      new_score;

  uint8_t prediction =
      argmaxN(
          fault_scores,
          OUTPUT_SIZE
      );

  uint32_t inference_us =
      micros() - start_us;

  sendForce(
      fault_scores,
      prediction,
      inference_us,
      class_id,
      old_score,
      new_score
  );
}
