import path from "path";
import fs from "fs";

import { expect } from "chai";
import * as tf from "@tensorflow/tfjs-node";

import models from "./index";

interface Image {
  image: string;
  test: (expect: any, results: any) => void;
}

interface Model {
  description: string;
  model: string;
  images: Image[];
}

const FIXTURES_PATH = path.join(__dirname, "..", "..", "..", "fixtures");
const OBJECT_DETECTION = path.join(FIXTURES_PATH, "object-detection.js");
const CLASSIFICATION = path.join(FIXTURES_PATH, "classification.js");

const objectDetectionModels = require(OBJECT_DETECTION) as Model[];
const classificationModels = require(CLASSIFICATION) as Model[];

function infersProperType(m: Model, type: string) {
  it("infers proper type", async () => {
    const modelPath = path.join(FIXTURES_PATH, m.model);
    const model = await models.load(modelPath);
    expect(model.type).to.equal(type);
  });
}

function shouldNotLeak(m: Model, detect: "detect" | "classify") {
  it(`${detect} should not leak`, async () => {
    const modelPath = path.join(FIXTURES_PATH, m.model);
    const model = await models.load(modelPath);

    const numOfTensorsBefore = tf.memory().numTensors;

    for (const item of m.images) {
      const imagePath = path.join(FIXTURES_PATH, item.image);
      const image = fs.readFileSync(imagePath);
      await model[detect](image);
    }

    expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
  });
}

function shouldGenerateOutput(m: Model, detect: "detect" | "classify") {
  it(`${detect} should generate output`, async () => {
    const modelPath = path.join(FIXTURES_PATH, m.model);
    const model = await models.load(modelPath);

    for (const item of m.images) {
      const imagePath = path.join(FIXTURES_PATH, item.image);
      const image = fs.readFileSync(imagePath);

      const results = await model[detect](image);

      item.test(expect, results);
    }
  });
}

objectDetectionModels.map((m) => {
  describe(`Object Detection ${m.description}`, () => {
    infersProperType(m, "detection");
    shouldNotLeak(m, "detect");
    shouldGenerateOutput(m, "detect");

    it("detect should generate output with options", async () => {
      const modelPath = path.join(FIXTURES_PATH, m.model);
      const model = await models.load(modelPath);

      const imagePath = path.join(FIXTURES_PATH, m.images[0].image);
      const image = fs.readFileSync(imagePath);
      const results = await model.detect(image, {
        maxNumberOfBoxes: 1,
        scoreThreshold: 0,
      });

      expect(results).to.have.lengthOf(1);
    });
  });
});

classificationModels.map((m) => {
  describe(`Classification ${m.description}`, () => {
    infersProperType(m, "classification");
    shouldNotLeak(m, "classify");
    shouldGenerateOutput(m, "classify");
  });
});
