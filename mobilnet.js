const imageURL = "cat.jpg";
const GOOGLE_CLOUD_STORAGE_DIR =
  "https://storage.googleapis.com/tfjs-models/savedmodel/";
const MODEL_FILE_URL = "mobilenet_v2_1.0_224/model.json";
const INPUT_NODE_NAME = "images";
const OUTPUT_NODE_NAME = "module_apply_default/MobilenetV2/Logits/output";
const PREPROCESS_DIVISOR = tf.scalar(255 / 2);

const PATH_MODEL = "my_model/model.json"

const cat = document.getElementById("cat");
const resultElement = document.getElementById("result");

let model = null;

cat.onload = async () => {
  resultElement.innerText = "Loading MobileNet...";

  // model = await tf.loadGraphModel(GOOGLE_CLOUD_STORAGE_DIR + MODEL_FILE_URL);
  model = await tf.loadLayersModel(PATH_MODEL);

  console.log("model", model.summary());

  const pixels = tf.browser.fromPixels(cat);

  // model.predict(pixels)

  let result = predict(pixels);

  console.log("result", result);

  // const topK = getTopKClasses(result, 5);
  // console.timeEnd("First prediction");

  // resultElement.innerText = "";
  // topK.forEach((x) => {
  //   resultElement.innerText += `${x.value.toFixed(3)}: ${x.label}\n`;
  // });

  // model.dispose();
};

const predict = (input) => {
  const preprocessedInput = tf.div(
    tf.sub(input.asType("float32"), PREPROCESS_DIVISOR),
    PREPROCESS_DIVISOR
  );

  const t = tf.reshape(preprocessedInput, [1, 180,180,3])
  console.log("t", t);

  return model.predict(t)

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

cat.src = imageURL;
