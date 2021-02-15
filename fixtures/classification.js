const test = (label) => (expect, results) => {
  expect(results[0].label).to.equal(label);
};

module.exports = [
  {
    description: "v1.2.x",
    model: "classification-model-v1.2.x",
    images: [
      { image: "cat.jpeg", test: test("cat") },
      { image: "reindeer.jpeg", test: test("reindeer") },
    ],
  },
  {
    description: "v1.3.x",
    model: "classification-model-v1.3.x",
    images: [
      { image: "cat.jpeg", test: test("cat") },
      { image: "reindeer.jpeg", test: test("reindeer") },
    ],
  },
  {
    description: "png",
    model: "classification-model-v1.3.x",
    images: [{ image: "image.png", test: test("none") }],
  },
  {
    description: "vx.x.x",
    model: "classification-notebook-custom",
    images: [
      { image: "cat.jpeg", test: test("Cork") },
      { image: "reindeer.jpeg", test: test("Cork") },
    ],
  },
];
