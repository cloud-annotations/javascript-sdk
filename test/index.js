const loadImage = async src => {
  const x = new Image();
  return await new Promise((resolve, _) => {
    x.onload = () => resolve(x);
    x.src = src;
  });
};

describe("Object Detection", () => {
  it("infers proper type", async () => {
    const model = await models.load("/test/object_detection_model");
    chai.expect(model.type).to.equal("detection");
  });

  it("detect should not leak", async () => {
    const model = await models.load("/test/object_detection_model");

    const numOfTensorsBefore = tf.memory().numTensors;

    const image = await loadImage("/test/image.jpg");
    await model.detect(image);

    chai.expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
  });

  it("detect should generate output", async () => {
    const model = await models.load("/test/object_detection_model");

    const image = await loadImage("/test/image.jpg");
    const results = await model.detect(image);

    chai.expect(results).to.have.lengthOf(3);
  });
});

describe("Classification", () => {
  it("infers proper type", async () => {
    const model = await models.load("/test/classification_model");
    chai.expect(model.type).to.equal("classification");
  });

  it("classify should not leak", async () => {
    const model = await models.load("/test/classification_model");

    const numOfTensorsBefore = tf.memory().numTensors;

    const image = await loadImage("/test/image.jpg");
    await model.classify(image);

    chai.expect(tf.memory().numTensors).to.equal(numOfTensorsBefore);
  });

  it("classify should generate output", async () => {
    const model = await models.load("/test/classification_model");

    const cat = await loadImage("/test/cat.jpeg");
    const reindeer = await loadImage("/test/reindeer.jpeg");
    const catResults = await model.classify(cat);
    const reindeerResults = await model.classify(reindeer);

    chai.expect(catResults[0].label).to.equal("cat");
    chai.expect(reindeerResults[0].label).to.equal("reindeer");
  });
});
