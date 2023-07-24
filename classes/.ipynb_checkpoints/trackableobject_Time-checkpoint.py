class TrackableObject:
	def __init__(self, objectID, centroid, region):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
        self.regions = [region]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False