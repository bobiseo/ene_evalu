from model.randomforest import RandomForest

# Model prediction function
def model_predict(data, df, name):
    print(f"🔍 Running model: {name}")

    # 1. Instantiate model object
    model = RandomForest(name, data.get_embeddings(), data.get_type())

    # 2. Train
    model.train(data)

    # 3. Predict
    preds = model.predict(data.X_test)

    # 4. Print evaluation results
    model.print_results(data)

    # 5. Return prediction results
    return preds

# Evaluation function (if needed)
def model_evaluate(model, data):
    model.print_results(data)
