module.exports = function (config) {
  config.set({
    client: {
      mocha: {
        timeout: 10_000, // 10 seconds - upped from 2 seconds
      },
    },
    frameworks: ["mocha", "chai", "karma-typescript"],
    files: [
      { pattern: "./node_modules/@tensorflow/tfjs/dist/tf.min.js" },
      { pattern: "./src/**/*.ts" },
      {
        pattern: "./../../fixtures/**/*",
        included: false,
        served: true,
      },
    ],
    preprocessors: {
      "**/*.ts": "karma-typescript", // *.tsx for React Jsx
    },
    karmaTypescriptConfig: {
      tsconfig: "tsconfig.json",
    },
    reporters: ["progress", "karma-typescript"],
    browsers: ["Chrome"],
  });
};
