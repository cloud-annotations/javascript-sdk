import { promises as fs } from "fs";
import * as tf from "@tensorflow/tfjs-node";
import core from "@cloud-annotations/core";

export * from "@cloud-annotations/core";

const models = {
  load: async (path: string) => {
    const graphPath = path + "/model.json";
    const labelsPath = path + "/labels.json";
    const handler = tf.io.fileSystem(graphPath);
    const graphPromise = tf.loadGraphModel(handler);
    const labelsPromise = await fs.readFile(labelsPath, "utf8");
    const [graph, labels] = await Promise.all([graphPromise, labelsPromise]);

    return core._init(tf, graph, JSON.parse(labels));
  },
};

module.exports = models;
export default models;
