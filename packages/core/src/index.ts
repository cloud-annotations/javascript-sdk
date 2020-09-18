const TYPE_DETECTION = "detection";
const TYPE_CLASSIFICATION = "classification";

const calculateMaxScores = (scores: any, numBoxes: any, numClasses: any) => {
  const maxes = [];
  const classes = [];
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE;
    let index = -1;
    for (let j = 0; j < numClasses; j++) {
      if (scores[i * numClasses + j] > max) {
        max = scores[i * numClasses + j];
        index = j;
      }
    }
    maxes[i] = max;
    classes[i] = index;
  }
  return [maxes, classes];
};

const buildDetectedObjects = (
  width: any,
  height: any,
  boxes: any,
  scores: any,
  indexes: any,
  classes: any,
  labels: any
) => {
  const count = indexes.length;
  const objects = [];
  for (let i = 0; i < count; i++) {
    const bbox = [];
    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j];
    }
    const minY = bbox[0] * height;
    const minX = bbox[1] * width;
    const maxY = bbox[2] * height;
    const maxX = bbox[3] * width;
    bbox[0] = minX;
    bbox[1] = minY;
    bbox[2] = maxX - minX;
    bbox[3] = maxY - minY;
    objects.push({
      bbox: bbox,
      class: labels[parseInt(classes[indexes[i]])], // deprecate.
      label: labels[parseInt(classes[indexes[i]])],
      score: scores[indexes[i]],
    });
  }
  return objects;
};

const runObjectDetectionPrediction = async (
  tf: any,
  graph: any,
  labels: any,
  input: any,
  { maxNumberOfBoxes = 20, iouThreshold = 0.5, scoreThreshold = 0.5 } = {}
) => {
  const batched = tf.tidy(() => {
    let img;
    try {
      img = tf.browser.fromPixels(input);
    } catch {
      img = tf.node.decodeImage(input, 3);
    }
    // Reshape to a single-element batch so we can pass it to executeAsync.
    return img.expandDims(0);
  });

  const height = batched.shape[1];
  const width = batched.shape[2];

  const result = await graph.executeAsync(batched);

  const scores = result[0].dataSync();
  const boxes = result[1].dataSync();

  // clean the webgl tensors
  batched.dispose();
  tf.dispose(result);

  const [maxScores, classes] = calculateMaxScores(
    scores,
    result[0].shape[1],
    result[0].shape[2]
  );

  const prevBackend = tf.getBackend();
  // run post process in cpu
  if (prevBackend !== "tensorflow") {
    tf.setBackend("cpu");
  }
  try {
    const indexTensor = tf.tidy(() => {
      const boxes2 = tf.tensor2d(boxes, [
        result[1].shape[1],
        result[1].shape[3],
      ]);

      return tf.image.nonMaxSuppression(
        boxes2,
        maxScores,
        maxNumberOfBoxes,
        iouThreshold,
        scoreThreshold
      );
    });
    const indexes = indexTensor.dataSync();
    indexTensor.dispose();
    // restore previous backend
    tf.setBackend(prevBackend);

    return buildDetectedObjects(
      width,
      height,
      boxes,
      maxScores,
      indexes,
      classes,
      labels
    );
  } catch (e) {
    tf.setBackend(prevBackend);
    throw e;
  }
};

const runClassificationPrediction = async (
  tf: any,
  graph: any,
  labels: any,
  input: any,
  options = {}
) => {
  if (options) {
    // no op
  }
  const batched = tf.tidy(() => {
    let img;
    try {
      img = tf.browser.fromPixels(input);
    } catch {
      img = tf.node.decodeImage(input, 3);
    }
    const small = tf.image.resizeBilinear(img, [224, 224]).div(255);

    // Reshape to a single-element batch so we can pass it to executeAsync.
    return small.expandDims(0).toFloat();
  });

  const results = graph.execute({ Placeholder: batched });

  const scores = results.arraySync()[0];

  results.dispose();
  batched.dispose();

  const finalScores = scores.map((score: any, i: any) => ({
    label: labels[i],
    score: score,
  }));

  finalScores.sort((a: any, b: any) => b.score - a.score);

  return finalScores;
};

const CONTROL_FLOW_OPS = ["Switch", "Merge", "Enter", "Exit", "NextIteration"];
// const DYNAMIC_SHAPE_OPS = [
//   "NonMaxSuppressionV2",
//   "NonMaxSuppressionV3",
//   "Where",
// ];

const checkControlFlow = (graph: any) => {
  return (
    Object.values(graph.executor.graph.nodes).find((n: any) =>
      CONTROL_FLOW_OPS.includes(n.op)
    ) !== undefined
  );
};

export default {
  _init: async (tf: any, graph: any, labels: any) => {
    const hasControlFlowOps = checkControlFlow(graph);

    // If there are control flow ops it's probably object detection.
    if (hasControlFlowOps) {
      return {
        type: TYPE_DETECTION,
        detect: async (input: any, options?: any) =>
          await runObjectDetectionPrediction(tf, graph, labels, input, options),
        classify: async () => {},
      };
    }

    // Otherwise, probably classification.
    return {
      type: TYPE_CLASSIFICATION,
      classify: async (input: any, options?: any) =>
        await runClassificationPrediction(tf, graph, labels, input, options),
      detect: async () => {},
    };
  },
};
