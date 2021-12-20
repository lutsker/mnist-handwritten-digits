from model.fcnn import model_factory
from data import dataset
from trainer import trainer

model = model_factory()
print(model.summary())
data = dataset()

train = trainer(model, data)
model = train(epochs=5, batch_size=64)
model.save("model/saved_models/one-epoch-fcnn")

#(x, y) = data('test')
#score = model.evaluate(x, y, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])
