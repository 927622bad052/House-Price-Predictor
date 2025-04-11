import unittest
from house_price_predictor import load_data, train_model, predict

class TestHousePricePredictor(unittest.TestCase):
    def test_load_data(self):
        data = load_data()
        self.assertEqual(data.shape, (5, 3))

    def test_train_model(self):
        model = train_model()
        self.assertIsNotNone(model)

    def test_predict(self):
        model = train_model()
        prediction = predict(model, 3, 850)
        self.assertTrue(prediction[0] > 0)

if __name__ == "__main__":
    unittest.main()
