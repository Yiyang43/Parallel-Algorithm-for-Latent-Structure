from IL2HGraphFinder import IL2HGraphFinder
import pandas as pd
import misc as M
import json


if __name__ == "__main__":
    with open("params.json") as f:
        PARAMS = json.load(f)

    model = IL2HGraphFinder(
        alpha=PARAMS["alpha"],
        maxk=PARAMS["maxk"],
        sample=PARAMS["sample"],
        n=PARAMS["n_samples"],
    )

    if PARAMS["scenario"] is not None:
        model.runScenario(scenario=PARAMS["scenario"], stage=PARAMS["stage"])
    else:
        df = pd.read_csv(PARAMS["data_path"])
        model.runSample(df, stage=PARAMS["stage"])
