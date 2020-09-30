import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import sucrase from "@rollup/plugin-sucrase";
import { terser } from "rollup-plugin-terser";

import pkg from "./package.json";

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
    output: [
      { file: pkg.main, format: "cjs" },
      { file: pkg.module, format: "es" },
    ],
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
