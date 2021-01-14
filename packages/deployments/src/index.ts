import { ReadStream } from "fs";
import { BaseService, IamAuthenticator } from "ibm-cloud-sdk-core";

import readImage from "./utils/read-image";

namespace Deployments {
  export interface Options {
    auth: string | undefined;
    region: string;
    version?: string;
  }

  export interface PredictOptions {
    deployment: string;
    image: ReadStream;
  }
}

export class Deployments extends BaseService {
  version: string;
  url: string;

  constructor(options: Deployments.Options) {
    const authenticator = new IamAuthenticator({
      apikey: options.auth!,
    });

    super({ authenticator });

    this.url = `https://${options.region}.ml.cloud.ibm.com/ml`;
    this.version = options.version ?? "2021-01-14";
  }

  public async predict(options: Deployments.PredictOptions) {
    const base64image = await readImage(options.image);

    const res = await this.createRequest({
      defaultOptions: {
        serviceUrl: this.url,
      },
      options: {
        url: `/v4/deployments/${options.deployment}/predictions`,
        method: "POST",
        qs: {
          version: this.version,
        },
        body: {
          input_data: [
            {
              values: [base64image],
            },
          ],
        },
      },
    });

    res.data = res.result.predictions[0].values[0].scores;

    delete res.result;

    return res;
  }
}
