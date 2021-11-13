

def trainer(model, data):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    (x, y) = data('train')
    def result(epochs, batch_size):
        model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        return model 
    return result

