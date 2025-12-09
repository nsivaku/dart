class BaseModel():
    def __init__(self, id=None, model_name=None):
        self.id = id
        self.model_name = model_name

    def run(self, images, queries):
        """
        Runs the model on the given input.

        Args:
            images (list of PIL.Image or list of str): A list of images to be processed. 
                Each item in the list can be a PIL Image object or a filepath (str) to an image.
            queries (list of str): Strings representing the query to be executed on the images.
        """        
        raise NotImplementedError