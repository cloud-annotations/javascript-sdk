const test = (length) => (expect, results) => {
  expect(results).to.have.lengthOf(length);
};

module.exports = [
  // {
  //   description: "test for experimental tf2 model",
  //   model: "tf2-trial",
  //   images: [{ image: "image.jpg", test: test(3) }],
  // },
  {
    description: "v1.2.x",
    model: "object-detection-model-v1.2.x",
    images: [{ image: "image.jpg", test: test(3) }],
  },
  {
    description: "v1.3.x",
    model: "object-detection-model-v1.3.x",
    images: [{ image: "image.jpg", test: test(3) }],
  },
  // {
  //   description: "v1.3.x - apr 7 issue",
  //   model: "error-apr-7",
  //   images: [{ image: "image.jpg", test: test(3) }],
  // },
  // {
  //   description: "v1.3.x - apr 20 issue",
  //   model: "error-apr-20",
  //   images: [{ image: "image.jpg", test: test(3) }],
  // },
  {
    description: "v1.3.2 - 2 class",
    model: "2-class-v1.3.2",
    images: [{ image: "image.jpg", test: test(2) }],
  },
  {
    description: "v1.3.2 - many classes",
    model: "many-classes-v1.3.2",
    images: [{ image: "image.jpg", test: test(0) }],
  },
  // {
  //   description: "v1.3.2 - 8 objects",
  //   model: "8-objects",
  //   images: [{ image: "image.jpg", test: test(3) }],
  // },
  {
    description: "v1.3.2 - empty class",
    model: "empty-class",
    images: [{ image: "image.jpg", test: test(0) }],
  },
  {
    description: "png",
    model: "object-detection-model-v1.3.x",
    images: [{ image: "image.png", test: test(0) }],
  },
];
