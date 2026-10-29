/*
  ============================================================
  TINY_CONV 12 CLASSES - F2 DYNAMIQUE - COMPATIBILITE TFLM
  Arduino Nano 33 BLE + ancienne API TensorFlow Lite Micro

  IMPORTANT :
  Le modele principal doit etre :
      f2_model_original_with_logits.tflite
  converti en :
      model_data.h

  Ce modele expose :
      output(0) = Softmax final [1,12] INT8
      output(1) = logits FullyConnected [1,12] INT8

  Cela evite l'API MicroInterpreter::tensor()/GetTensor()
  absente de certaines bibliotheques Arduino TFLM.

  COMMANDES PC -> Nano :

  0xA5 + 1960 octets
      inference nominale

  0xA6 + index + bit + 1960 octets
      F2 bit-flip transitoire sur un logit

  0xA7 + index + reserve + 1960 octets
      F2 mise a zero quantifiee sur un logit

  REPONSES :

  0x5A + 12 scores + prediction + temps
      nominal

  0x5B + 12 scores + prediction + temps + old + new
      F2
  ============================================================
*/

#include <TensorFlowLite.h>

#include "model_data.h"
#include "softmax_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ============================================================
// IDENTIFICATION
// ============================================================

const char* MODEL_NAME =
    "TINY_CONV ORIGINAL - F2 FC LOGITS - COMPAT";


// ============================================================
// PROTOCOLE
// ============================================================

constexpr uint8_t CMD_NOMINAL    = 0xA5;
constexpr uint8_t CMD_F2_BITFLIP = 0xA6;
constexpr uint8_t CMD_F2_ZERO    = 0xA7;

constexpr uint8_t RESP_NOMINAL   = 0x5A;
constexpr uint8_t RESP_F2        = 0x5B;

constexpr uint8_t ERROR_PARAM    = 0xED;
constexpr uint8_t ERROR_RX       = 0xEE;
constexpr uint8_t ERROR_INVOKE   = 0xEF;

constexpr int INPUT_SIZE  = 1960;
constexpr int OUTPUT_SIZE = 12;


// ============================================================
// TENSOR ARENAS
// ============================================================

constexpr int kMainArenaSize    = 16 * 1024;
constexpr int kSoftmaxArenaSize = 16 * 1024;


// ============================================================
// TFLM
// ============================================================

namespace {

const tflite::Model* main_model = nullptr;
const tflite::Model* softmax_model = nullptr;

tflite::MicroInterpreter* main_interpreter = nullptr;
tflite::MicroInterpreter* softmax_interpreter = nullptr;

TfLiteTensor* main_input = nullptr;

// output(0) du modele instrumente
TfLiteTensor* main_output = nullptr;

// output(1) du modele instrumente = tensor logits FC
TfLiteTensor* main_logits = nullptr;

TfLiteTensor* softmax_input = nullptr;
TfLiteTensor* softmax_output = nullptr;

alignas(16) uint8_t main_arena[kMainArenaSize];
alignas(16) uint8_t softmax_arena[kSoftmaxArenaSize];

}  // namespace


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
    uint32_t timeout_ms = 3000
) {

  size_t received = 0;
  uint32_t last_receive = millis();

  while (received < length) {

    int availableBytes =
        Serial.available();

    if (availableBytes > 0) {

      size_t remaining =
          length - received;

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

    if (
        millis() - last_receive
        > timeout_ms
    ) {
      return false;
    }
  }

  return true;
}


uint8_t argmax12(
    const int8_t* scores
) {

  uint8_t prediction = 0;
  int8_t best_score = scores[0];

  for (int i = 1; i < OUTPUT_SIZE; i++) {

    if (scores[i] > best_score) {
      best_score = scores[i];
      prediction =
          static_cast<uint8_t>(i);
    }
  }

  return prediction;
}


void sendNominalResponse(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us
) {

  Serial.write(RESP_NOMINAL);

  Serial.write(
      reinterpret_cast<const uint8_t*>(
          scores
      ),
      OUTPUT_SIZE
  );

  Serial.write(prediction);

  Serial.write(
      reinterpret_cast<uint8_t*>(
          &inference_us
      ),
      sizeof(inference_us)
  );

  Serial.flush();
}


void sendF2Response(
    const int8_t* scores,
    uint8_t prediction,
    uint32_t inference_us,
    int8_t old_value,
    int8_t new_value
) {

  Serial.write(RESP_F2);

  Serial.write(
      reinterpret_cast<const uint8_t*>(
          scores
      ),
      OUTPUT_SIZE
  );

  Serial.write(prediction);

  Serial.write(
      reinterpret_cast<uint8_t*>(
          &inference_us
      ),
      sizeof(inference_us)
  );

  Serial.write(
      reinterpret_cast<uint8_t*>(
          &old_value
      ),
      1
  );

  Serial.write(
      reinterpret_cast<uint8_t*>(
          &new_value
      ),
      1
  );

  Serial.flush();
}


// ============================================================
// INFORMATIONS
// ============================================================

void printInfo() {

  Serial.println();
  Serial.println("========================================");
  Serial.println("  TINY_CONV - F2 DYNAMIQUE - COMPAT");
  Serial.println("========================================");

  Serial.print("Modele : ");
  Serial.println(MODEL_NAME);

  Serial.println();
  Serial.print("Main model : ");
  Serial.print(g_model_len);
  Serial.println(" octets");

  Serial.print("Softmax tail : ");
  Serial.print(g_softmax_model_len);
  Serial.println(" octets");

  Serial.println();
  Serial.println("Main output(0) : Softmax [1,12]");
  Serial.println("Main output(1) : Logits FC [1,12]");

  Serial.println();
  Serial.print("Main arena utilisee : ");
  Serial.println(
      static_cast<int>(
          main_interpreter->arena_used_bytes()
      )
  );

  Serial.print("Main arena reservee : ");
  Serial.println(kMainArenaSize);

  Serial.print("Softmax arena utilisee : ");
  Serial.println(
      static_cast<int>(
          softmax_interpreter->arena_used_bytes()
      )
  );

  Serial.print("Softmax arena reservee : ");
  Serial.println(kSoftmaxArenaSize);

  Serial.println();
  Serial.println("Commandes :");
  Serial.println("  0xA5 = nominal");
  Serial.println("  0xA6 = F2 bitflip logit");
  Serial.println("  0xA7 = F2 zero logit");

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

  Serial.setTimeout(3000);


  // ----------------------------------------------------------
  // MODELE PRINCIPAL
  // ----------------------------------------------------------

  main_model =
      tflite::GetModel(g_model);

  if (main_model == nullptr) {
    fatalError("MAIN_MODEL_NULL");
  }

  if (
      main_model->version()
      != TFLITE_SCHEMA_VERSION
  ) {
    fatalError("MAIN_BAD_SCHEMA");
  }


  static tflite::MicroMutableOpResolver<4>
      main_resolver;

  if (
      main_resolver.AddConv2D()
      != kTfLiteOk
  ) {
    fatalError("CONV2D");
  }

  if (
      main_resolver.AddFullyConnected()
      != kTfLiteOk
  ) {
    fatalError("FULLY_CONNECTED");
  }

  if (
      main_resolver.AddSoftmax()
      != kTfLiteOk
  ) {
    fatalError("SOFTMAX");
  }

  if (
      main_resolver.AddReshape()
      != kTfLiteOk
  ) {
    fatalError("RESHAPE");
  }


  // MEME FORME DE CONSTRUCTEUR QUE TON UART ORIGINAL.
  static tflite::MicroInterpreter
      static_main_interpreter(
          main_model,
          main_resolver,
          main_arena,
          kMainArenaSize,
          nullptr,
          nullptr
      );

  main_interpreter =
      &static_main_interpreter;


  if (
      main_interpreter->AllocateTensors()
      != kTfLiteOk
  ) {
    fatalError("MAIN_ALLOCATE");
  }


  main_input =
      main_interpreter->input(0);

  main_output =
      main_interpreter->output(0);

  main_logits =
      main_interpreter->output(1);


  if (
      main_input == nullptr
      || main_output == nullptr
      || main_logits == nullptr
  ) {
    fatalError("MAIN_TENSOR_NULL");
  }


  if (
      main_input->type != kTfLiteInt8
      || main_input->dims->size != 2
      || main_input->dims->data[0] != 1
      || main_input->dims->data[1] != INPUT_SIZE
  ) {
    fatalError("BAD_INPUT");
  }


  if (
      main_output->type != kTfLiteInt8
      || main_output->dims->size != 2
      || main_output->dims->data[0] != 1
      || main_output->dims->data[1] != OUTPUT_SIZE
  ) {
    fatalError("BAD_OUTPUT0");
  }


  if (
      main_logits->type != kTfLiteInt8
      || main_logits->dims->size != 2
      || main_logits->dims->data[0] != 1
      || main_logits->dims->data[1] != OUTPUT_SIZE
  ) {
    fatalError("BAD_OUTPUT1");
  }


  // ----------------------------------------------------------
  // SOFTMAX TAIL
  // ----------------------------------------------------------

  softmax_model =
      tflite::GetModel(
          g_softmax_model
      );

  if (softmax_model == nullptr) {
    fatalError("SOFTMAX_MODEL_NULL");
  }

  if (
      softmax_model->version()
      != TFLITE_SCHEMA_VERSION
  ) {
    fatalError("SOFTMAX_BAD_SCHEMA");
  }


  static tflite::MicroMutableOpResolver<1>
      softmax_resolver;

  if (
      softmax_resolver.AddSoftmax()
      != kTfLiteOk
  ) {
    fatalError("TAIL_SOFTMAX");
  }


  // MEME FORME DE CONSTRUCTEUR QUE TON UART ORIGINAL.
  static tflite::MicroInterpreter
      static_softmax_interpreter(
          softmax_model,
          softmax_resolver,
          softmax_arena,
          kSoftmaxArenaSize,
          nullptr,
          nullptr
      );

  softmax_interpreter =
      &static_softmax_interpreter;


  if (
      softmax_interpreter->AllocateTensors()
      != kTfLiteOk
  ) {
    fatalError("SOFTMAX_ALLOCATE");
  }


  softmax_input =
      softmax_interpreter->input(0);

  softmax_output =
      softmax_interpreter->output(0);


  if (
      softmax_input == nullptr
      || softmax_output == nullptr
  ) {
    fatalError("SOFTMAX_TENSOR_NULL");
  }


  if (
      softmax_input->type != kTfLiteInt8
      || softmax_input->dims->size != 2
      || softmax_input->dims->data[0] != 1
      || softmax_input->dims->data[1] != OUTPUT_SIZE
  ) {
    fatalError("SOFTMAX_BAD_INPUT");
  }


  printInfo();

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


  uint8_t command =
      Serial.read();


  if (
      command != CMD_NOMINAL
      && command != CMD_F2_BITFLIP
      && command != CMD_F2_ZERO
  ) {
    return;
  }


  // ----------------------------------------------------------
  // PARAMETRES F2
  // ----------------------------------------------------------

  uint8_t activation_index = 0;
  uint8_t bit = 0;


  if (
      command == CMD_F2_BITFLIP
      || command == CMD_F2_ZERO
  ) {

    uint8_t params[2];

    if (
        !readExactBytes(
            params,
            2,
            3000
        )
    ) {
      Serial.write(ERROR_RX);
      Serial.flush();
      return;
    }


    activation_index = params[0];
    bit = params[1];


    if (
        activation_index >= OUTPUT_SIZE
    ) {
      Serial.write(ERROR_PARAM);
      Serial.flush();
      return;
    }


    if (
        command == CMD_F2_BITFLIP
        && bit >= 8
    ) {
      Serial.write(ERROR_PARAM);
      Serial.flush();
      return;
    }
  }


  // ----------------------------------------------------------
  // ENTREE 1960 INT8
  // ----------------------------------------------------------

  if (
      !readExactBytes(
          reinterpret_cast<uint8_t*>(
              main_input->data.int8
          ),
          INPUT_SIZE,
          3000
      )
  ) {
    Serial.write(ERROR_RX);
    Serial.flush();
    return;
  }


  // ----------------------------------------------------------
  // INFERENCE COMPLETE
  // ----------------------------------------------------------

  uint32_t start_us =
      micros();


  if (
      main_interpreter->Invoke()
      != kTfLiteOk
  ) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }


  // ----------------------------------------------------------
  // MODE NOMINAL
  // ----------------------------------------------------------

  if (
      command == CMD_NOMINAL
  ) {

    uint32_t inference_us =
        micros() - start_us;


    uint8_t prediction =
        argmax12(
            main_output->data.int8
        );


    sendNominalResponse(
        main_output->data.int8,
        prediction,
        inference_us
    );

    return;
  }


  // ----------------------------------------------------------
  // COPIE DES LOGITS FC EXPOSES COMME output(1)
  // ----------------------------------------------------------

  for (
      int i = 0;
      i < OUTPUT_SIZE;
      i++
  ) {

    softmax_input->data.int8[i] =
        main_logits->data.int8[i];
  }


  // ----------------------------------------------------------
  // F2 TRANSITOIRE SUR LA COPIE
  // ----------------------------------------------------------

  int8_t old_value =
      softmax_input->data.int8[
          activation_index
      ];

  int8_t new_value =
      old_value;


  if (
      command == CMD_F2_BITFLIP
  ) {

    uint8_t* raw =
        reinterpret_cast<uint8_t*>(
            softmax_input->data.int8
        );

    raw[activation_index] ^=
        static_cast<uint8_t>(
            1u << bit
        );


    new_value =
        softmax_input->data.int8[
            activation_index
        ];
  }
  else {

    // Valeur reelle 0 en quantification INT8
    // = zero_point du tenseur logits.
    new_value =
        static_cast<int8_t>(
            softmax_input->params.zero_point
        );

    softmax_input->data.int8[
        activation_index
    ] =
        new_value;
  }


  // ----------------------------------------------------------
  // SOFTMAX SUR LES LOGITS FAUTES
  // ----------------------------------------------------------

  if (
      softmax_interpreter->Invoke()
      != kTfLiteOk
  ) {
    Serial.write(ERROR_INVOKE);
    Serial.flush();
    return;
  }


  uint32_t inference_us =
      micros() - start_us;


  uint8_t prediction =
      argmax12(
          softmax_output->data.int8
      );


  sendF2Response(
      softmax_output->data.int8,
      prediction,
      inference_us,
      old_value,
      new_value
  );
}
