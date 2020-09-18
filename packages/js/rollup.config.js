import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import sucrase from "@rollup/plugin-sucrase";
import { terser } from "rollup-plugin-terser";

export default [
  {
    input: "src/index.ts",
    plugins: [
      resolve(),
      commonjs(),
      sucrase({
        // so Rollup can convert TypeScript to JavaScript
        exclude: ["node_modules/**", "**/?(*.)test.ts"],
        transforms: ["typescript"],
      }),
    ],
    output: {
      format: "umd",
      name: "models",
      file: "dist/models.js",
    },
  },
  {
    input: "src/index.ts",
    plugins: [
      resolve(),
      commonjs(),
      sucrase({
        // so Rollup can convert TypeScript to JavaScript
        exclude: ["node_modules/**", "**/?(*.)test.ts"],
        transforms: ["typescript"],
      }),
      terser(),
    ],
    output: {
      format: "umd",
      name: "models",
      file: "dist/models.min.js",
    },
  },
];
