import { ReadStream } from "fs";

function readImage(stream: ReadStream): Promise<string> {
  return new Promise((resolve) => {
    stream.on("readable", () => {
      const buffer = stream.read();
      if (buffer) {
        resolve(buffer.toString("base64"));
      }
    });
  });
}

export default readImage;
