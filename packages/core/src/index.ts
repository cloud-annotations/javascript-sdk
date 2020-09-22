import * as _tf from "@tensorflow/tfjs";
import { GraphModel, Rank, Tensor, backend_util } from "@tensorflow/tfjs";

export enum ModelType {
  Classification = "classification",
  Detection = "detection",
}

export interface DetectionOptions {
  maxNumberOfBoxes?: number;
  iouThreshold?: number;
  scoreThreshold?: number;
}

export interface DetectionResult {
  bbox: number[];
  class: string;
  label: string;
  score: number;
}

export interface DetectionModel {
  type: ModelType.Detection;
  detect: (
    input: any,
    options?: DetectionOptions
  ) => Promise<DetectionResult[]>;
  classify: (
    input: any,
    options?: ClassificationOptions
  ) => Promise<ClassificationResult[]>;
}

export interface ClassificationOptions {}

export interface ClassificationResult {
  label: string;
  score: number;
}

export interface ClassificationModel {
  type: ModelType.Classification;
  detect: (
    input: any,
    options?: DetectionOptions
  ) => Promise<DetectionResult[]>;
  classify: (
    input: any,
    options?: ClassificationOptions
  ) => Promise<ClassificationResult[]>;
}

const calculateMaxScores = (
  scores: backend_util.TypedArray,
  numBoxes: number,
  numClasses: number
) => {
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
  width: number,
  height: number,
  boxes: backend_util.TypedArray,
  scores: number[],
  indexes: backend_util.TypedArray,
  classes: number[],
  labels: string[]
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
      class: labels[classes[indexes[i]]], // deprecate.
      label: labels[classes[indexes[i]]],
      score: scores[indexes[i]],
    });
  }
  return objects;
};

const runObjectDetectionPrediction = async (
  tf: typeof _tf,
  graph: GraphModel,
  labels: string[],
  input: any,
  { maxNumberOfBoxes = 20, iouThreshold = 0.5, scoreThreshold = 0.5 } = {}
) => {
  const batched: Tensor<Rank.R4> = tf.tidy(() => {
    let img: Tensor<Rank.R3>;
    try {
      img = tf.browser.fromPixels(input);
    } catch {
      // @ts-ignore
      img = tf.node.decodeImage(input, 3);
    }
    // Reshape to a single-element batch so we can pass it to executeAsync.
    return img.expandDims(0);
  });

  const height = batched.shape[1];
  const width = batched.shape[2];

  const result = (await graph.executeAsync(batched)) as Tensor<Rank.R4>[];

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
  tf: typeof _tf,
  graph: GraphModel,
  labels: string[],
  input: any,
  _options = {}
) => {
  const batched = tf.tidy(() => {
    let img: Tensor<Rank.R3>;
    try {
      img = tf.browser.fromPixels(input);
    } catch {
      // @ts-ignore
      img = tf.node.decodeImage(input, 3);
    }
    const small = tf.image.resizeBilinear(img, [224, 224]).div(255);

    // Reshape to a single-element batch so we can pass it to executeAsync.
    return small.expandDims(0).toFloat();
  });

  const results = graph.execute({ Placeholder: batched }) as Tensor<Rank.R2>;

  const scores = results.arraySync()[0];

  results.dispose();
  batched.dispose();

  const finalScores = scores.map((score, i) => ({
    label: labels[i],
    score: score,
  }));

  finalScores.sort((a, b) => b.score - a.score);

  return finalScores;
};

const CONTROL_FLOW_OPS = ["Switch", "Merge", "Enter", "Exit", "NextIteration"];

const checkControlFlow = (graph: GraphModel) => {
  return (
    // @ts-ignore
    Object.values(graph.executor.graph.nodes).find((n: any) =>
      CONTROL_FLOW_OPS.includes(n.op)
    ) !== undefined
  );
};

export default {
  _init: async (
    tf: typeof _tf,
    graph: GraphModel,
    labels: string[]
  ): Promise<DetectionModel | ClassificationModel> => {
    const hasControlFlowOps = checkControlFlow(graph);

    // If there are control flow ops it's probably object detection.
    if (hasControlFlowOps) {
      return {
        type: ModelType.Detection,
        detect: async (input: any, options?: DetectionOptions) =>
          await runObjectDetectionPrediction(tf, graph, labels, input, options),
        classify: async (_input: any, _options?: ClassificationOptions) => {
          throw Error("use `model.detect`");
        },
      };
    }

    // Otherwise, probably classification.
    return {
      type: ModelType.Classification,
      classify: async (input: any, options?: ClassificationOptions) =>
        await runClassificationPrediction(tf, graph, labels, input, options),
      detect: async (_input: any, _options?: DetectionOptions) => {
        throw Error("use `model.classify`");
      },
    };
  },
};
