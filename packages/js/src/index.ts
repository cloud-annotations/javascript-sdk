import * as tf from "@tensorflow/tfjs";
import core from "@cloud-annotations/core";

export default {
  load: async (path: string) => {
    const graphPath = path + "/model.json";
    const labelsPath = path + "/labels.json";
    const graphPromise = tf.loadGraphModel(graphPath);
    const labelsPromise = fetch(labelsPath).then((data) => data.json());
    const [graph, labels] = await Promise.all([graphPromise, labelsPromise]);

    return core._init(tf, graph, labels);
  },
};
