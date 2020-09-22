import { expect } from "chai";
import * as tf from "@tensorflow/tfjs";

import models from "./index";
import { DetectionModel, ModelType } from "@cloud-annotations/core";

interface Image {
  image: string;
  test: (expect: any, results: any) => void;
}

interface Model {
  description: string;
  model: string;
  images: Image[];
}

const path = require("path");
const fixtures = path.join(__dirname, "./../../../fixtures");

const objectDetectionModels = require("./../../../fixtures/object-detection.js") as Model[];
const classificationModels = require("./../../../fixtures/classification.js") as Model[];

async function loadImage(src: string) {
  const x = new Image();
  return await new Promise((resolve, _) => {
    x.onload = () => resolve(x);
    x.src = src;
  });
}

function infersProperType(m: Model, type: ModelType) {
  it("infers proper type", async () => {
    const modelPath = path.join(fixtures, m.model);
    const model = await models.load(modelPath);
    expect(model.type).to.equal(type);
  });
}

function shouldNotLeak(m: Model, detect: "detect" | "classify") {
  it(`${detect} should not leak`, async () => {
    const modelPath = path.join(fixtures, m.model);
    const model = await models.load(modelPath);

    const numOfTensorsBefore = tf.memory().numTensors;

    for (const item of m.images) {
      const imagePath = path.join(fixtures, item.image);
      const image = await loadImage(imagePath);
      await model[detect](image);
    }

    expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
  });
}

function shouldGenerateOutput(m: Model, detect: "detect" | "classify") {
  it(`${detect} should generate output`, async () => {
    const modelPath = path.join(fixtures, m.model);
    const model = await models.load(modelPath);

    for (const item of m.images) {
      const imagePath = path.join(fixtures, item.image);
      const image = await loadImage(imagePath);
      const results = await model[detect](image);
      item.test(expect, results);
    }
  });
}

objectDetectionModels.map((m) => {
  describe(`Object Detection ${m.description}`, () => {
    infersProperType(m, ModelType.Detection);
    shouldNotLeak(m, "detect");
    shouldGenerateOutput(m, "detect");

    it("detect should generate output with options", async () => {
      const modelPath = path.join(fixtures, m.model);
      const model = (await models.load(modelPath)) as DetectionModel;

      const imagePath = path.join(fixtures, m.images[0].image);
      const image = await loadImage(imagePath);

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
    infersProperType(m, ModelType.Classification);
    shouldNotLeak(m, "classify");
    shouldGenerateOutput(m, "classify");
  });
});
