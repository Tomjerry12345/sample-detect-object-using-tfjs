const PATH_SEL_BASAL = "data-testing/karsinoma_sel_basal/"
const PATH_SEL_SKUAMOSA = "data-testing/karsinoma_sel_skuamosa/"
const imageURL = "ISIC_0025471.jpg";

const classes = 1

const testing_image = (classes === 0 ? PATH_SEL_BASAL : PATH_SEL_SKUAMOSA) + imageURL

const GOOGLE_CLOUD_STORAGE_DIR =
  "https://storage.googleapis.com/tfjs-models/savedmodel/";
const MODEL_FILE_URL = "mobilenet_v2_1.0_224/model.json";
const INPUT_NODE_NAME = "images";
const OUTPUT_NODE_NAME = "module_apply_default/MobilenetV2/Logits/output";
const PREPROCESS_DIVISOR = tf.scalar(255 / 2);

const PATH_MODEL = "my-model/model.json";

const cat = document.getElementById("cat");
const resultElement = document.getElementById("result");

let model = null;

const LABELS = {
  0: "Daisy",
  1: "Dandelion",
  2: "Roses",
  3: "Sunflowers",
  4: "Tulips",
};

cat.onload = async () => {
  resultElement.innerText = "Loading MobileNet...";

  model = await tf.loadLayersModel(PATH_MODEL);

  const pixels = tf.browser.fromPixels(cat);

  let result = predict(pixels);

  const axis =  1

  const prediction = Array.from(result.argMax(axis).dataSync())

  console.log("prediction", prediction)

  let txtPredict = ""

  if (prediction[0] === 0) {
    txtPredict = "Karsinoma Sel Basal"
  } else {
    txtPredict = "Karsinoma Sel Skuamosa"
  }

  resultElement.innerText = txtPredict

  model.dispose();
};

const predict = (input) => {
  // const preprocessedInput = tf.div(
  //   tf.sub(input.asType("float32"), PREPROCESS_DIVISOR),
  //   PREPROCESS_DIVISOR
  // );

  const imResize = input.resizeBilinear([180, 180])

  const expandimsImage = tf.expandDims(imResize)

  const t = expandimsImage.reshape([1, 180, 180, 3]);

  return model.predict(t);

  // return model.execute({ [INPUT_NODE_NAME]: reshapedInput }, OUTPUT_NODE_NAME);
};

const getTopKClasses = (logits, topK) => {
  const predictions = tf.tidy(() => {
    return tf.softmax(logits);
  });

  const values = predictions.dataSync();
  predictions.dispose();

  let predictionList = [];
  for (let i = 0; i < values.length; i++) {
    predictionList.push({ value: values[i], index: i });
  }
  predictionList = predictionList
    .sort((a, b) => {
      return b.value - a.value;
    })
    .slice(0, topK);

  return predictionList.map((x) => {
    return { label: IMAGENET_CLASSES[x.index], value: x.value };
  });
};

cat.src = testing_image;
