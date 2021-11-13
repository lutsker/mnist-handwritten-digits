from model import model_factory
from data import dataset
from trainer import trainer

model = model_factory()
print(model.summary())
data = dataset()

train = trainer(model, data)
model = train(epochs=10, batch_size=128)

(x, y) = data('test')
score = model.evaluate(x, y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
