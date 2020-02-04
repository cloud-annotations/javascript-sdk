import resolve from "rollup-plugin-node-resolve";
import commonjs from "rollup-plugin-commonjs";
import babel from "rollup-plugin-babel";
import { uglify } from "rollup-plugin-uglify";

export default [
  {
    input: "src/index.js",
    plugins: [
      resolve(),
      commonjs(),
      babel({
        exclude: "node_modules/**",
        runtimeHelpers: true
      })
    ],
    output: {
      format: "umd",
      name: "models",
      file: "dist/models.js"
    }
  },
  {
    input: "src/index.js",
    plugins: [
      resolve(),
      commonjs(),
      babel({
        exclude: "node_modules/**",
        runtimeHelpers: true
      }),
      uglify()
    ],
    output: {
      format: "umd",
      name: "models",
      file: "dist/models.min.js"
    }
  }
];
