const loadImage = async (src) => {
  const x = new Image();
  return await new Promise((resolve, _) => {
    x.onload = () => resolve(x);
    x.src = src;
  });
};

const objectDetectionModels = [
  // { version: "v1.2.x", path: "/test/object-detection-model-v1.2.x" },
  // { version: "v1.3.x", path: "/test/object-detection-model-v1.3.x" },
  // { version: "v1.3.x", path: "/test/error-apr-7" },
  // { version: "v1.3.x", path: "/test/error-apr-20" },
  // { version: "v1.3.2", path: "/test/2-class-v1.3.2" },
  // { version: "v1.3.2", path: "/test/many-classes-v1.3.2" },
  { version: "v1.3.2", path: "/test/8-objects" },
  { version: "v1.3.2", path: "/test/empty-class" },
];

const classificationModels = [
  { version: "v1.2.x", path: "/test/classification-model-v1.2.x" },
  { version: "v1.3.x", path: "/test/classification-model-v1.3.x" },
];

objectDetectionModels.map((m) => {
  describe(`Object Detection ${m.version}`, () => {
    it("infers proper type", async () => {
      const model = await models.load(m.path);
      chai.expect(model.type).to.equal("detection");
    });

    it("detect should not leak", async () => {
      const model = await models.load(m.path);

      const numOfTensorsBefore = tf.memory().numTensors;

      const image = await loadImage("/test/image.jpg");
      await model.detect(image);

      chai.expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
    });

    it("detect should generate output", async () => {
      const model = await models.load(m.path);

      const image = await loadImage("/test/image.jpg");
      const results = await model.detect(image);

      chai.expect(results).to.have.lengthOf(3);
    });

    it("detect should generate output with options", async () => {
      const model = await models.load(m.path);

      const image = await loadImage("/test/image.jpg");
      const results = await model.detect(image, {
        maxNumberOfBoxes: 1,
        scoreThreshold: 0,
      });

      chai.expect(results).to.have.lengthOf(1);
    });
  });
});

classificationModels.map((m) => {
  describe(`Classification ${m.version}`, () => {
    it("infers proper type", async () => {
      const model = await models.load(m.path);
      chai.expect(model.type).to.equal("classification");
    });

    it("classify should not leak", async () => {
      const model = await models.load(m.path);

      const numOfTensorsBefore = tf.memory().numTensors;

      const image = await loadImage("/test/image.jpg");
      await model.classify(image);

      chai.expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
    });

    it("classify should generate output", async () => {
      const model = await models.load(m.path);

      const cat = await loadImage("/test/cat.jpeg");
      const reindeer = await loadImage("/test/reindeer.jpeg");
      const catResults = await model.classify(cat);
      const reindeerResults = await model.classify(reindeer);

      chai.expect(catResults[0].label).to.equal("cat");
      chai.expect(reindeerResults[0].label).to.equal("reindeer");
    });
  });
});
