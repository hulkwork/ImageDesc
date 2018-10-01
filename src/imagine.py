import cv2
import os.path
from sklearn.cluster import KMeans
import numpy as np
import imutils
from utils import misc, deep
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))


class ImageDescription(object):
    IMAGE = None
    CLUSTERS = None
    COLORS = None
    LABELS = None

    def __init__(self, image_path, conf, verbose=1):
        self.conf = conf
        self.IMAGE = image_path
        self.CLUSTERS = self.conf.get("clusters", 4)
        self.output_dir = self.conf['output_dir']
        self.base = os.path.basename(self.IMAGE)
        self.filename, self.file_ext = os.path.splitext(self.base)
        self.img = cv2.imread(filename=self.IMAGE)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.verbose = verbose
        self.reporters = {}

    def dominant_colors(self):
        # reshaping to a list of pixels
        img = self.img.reshape((self.img.shape[0] * self.img.shape[1], 3))

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_
        self.reporters['dominant_colors'] = self.COLORS.astype(int).tolist()
        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def plot_histogram(self):
        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "dominant_colors", self.file_ext)),
            img=chart)

    def kmeans_segementation(self):
        """
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
        :return:
        """
        Z = self.img.copy().reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(data=Z, K=self.CLUSTERS, bestLabels=None, criteria=criteria, attempts=10,
                                        flags=cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((self.img.shape))
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "kmeans", self.file_ext)),
            img=res2)

    def detect_anime_face(self):
        cascade_file = self.conf.get("cascade_file")

        cascade = cv2.CascadeClassifier(cascade_file)
        image = self.img.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.02,
                                         minNeighbors=5,
                                         minSize=(5, 5))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        self.reporters['face_detected'] = {"number_face_detected": len(faces),
                                           "face_box": faces.tolist() if type(faces) is not tuple else faces}
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "anime_face", self.file_ext)),
            img=image)

    def detect_shape(self):
        # load the image and resize it to a smaller factor so that
        # the shapes can be approximated better
        image = self.img.copy()
        # convert image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        shapes = {}
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M["m00"] != 0.0 and M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                shape = misc.detect(c)
                # draw the contours and the name of the shape on the image
                c = c.astype("float")
                c = c.astype("int")
                color_contour = (0, 255, 0)
                if shape is not None:
                    shapes[shape] = shapes.get(shape, 0) + 1
                    color_contour = (0, 255, 255)
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                cv2.drawContours(image, [c], -1, color_contour, 4)

        self.reporters['shapes'] = shapes
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "shape", self.file_ext)),
            img=image)

    def detect_object(self):
        image, details = deep.yolo_detection(image=self.img.copy())
        self.reporters["object_detections"] = details
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "object", self.file_ext)),
            img=image)

    def gradient(self,k=5):
        gray = self.gray.copy()
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "laplacian", self.file_ext)),
            img=laplacian)

    def canny_edges(self):
        edges = cv2.Canny(self.img.copy(), 100, 200)
        cv2.imwrite(
            filename=os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "canny", self.file_ext)),
            img=edges)

    def write_report(self):
        path_report = os.path.join(self.output_dir, "%s_%s%s" % (self.filename, "report", ".json"))
        print(79 * '*')
        pprint(self.reporters)
        print(79 * '*')

    def process(self):
        self.dominant_colors()
        self.plot_histogram()
        self.kmeans_segementation()
        self.detect_anime_face()
        self.detect_shape()
        self.detect_object()
        self.canny_edges()
        self.gradient(k=self.conf.get("kernel_gradient", 4))
        if self.verbose:
            self.write_report()
