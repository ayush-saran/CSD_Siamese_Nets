

def test_results(siamese_model, testing_set):

	y_pred = []
	bounding_boxes = []

	for image in testing_set:

		result_array = siamese_model.predict(image)
		predicted_class = np.argmax(result_array)
		bounding_box = ##Get bounding box. You have the code for it right, just append it

		y_pred.append(predicted_class)
		bounding_boxes.append(bounding_box)


	return y_pred, bounding_boxes



