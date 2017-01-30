# Deep Learning Project on Cell Segmentation

**Introduction:**

Automatic analysis of microscopy cell images has a crucial importance in many biomedical research fields.
Understanding cells dynamics and properties is the basis in cancer research, new drugs developments and other fields of importance.
The first task when analyzing cells microscopy images is cell segmentation: separation of cells pixels from background pixels and the separation of one cell pixels from the other.

**Data:**

The purpose of this project is performing semantic cell segmentation between cells and background.
In this project we will work with microscopy images of [H1299](https://en.wikipedia.org/wiki/H1299) lung cancer cells, data courtesy of Alon Labs, Weizmann Institute of Science [1].

Inputs are 64x64 pixels gray level images. <br />
Outputs are 64x64 pixels binary (segmentation) images.

* Training set consists of 4815 images.
* Validation set consists of 467 images.
* Testing set consists of 478 images.

![Alt text](/cell.png?raw=true "Samples Example")

**Performance evaluation:**

Segmentation performance will be evaluated using Dice similarity index between the segmented image and its ground-truth binary image

`D  = \frac{2|X \bigcap Y|}{|X|+|Y|}` <br />
Where X is the binary estimated segmentation image and Y is the binary ground-truth image.

Main requirement is TensorFlow.

1. Training: python main.py --mode train
2. If you want to resume and continue training: python main.py --mode resume --checkpoint <CHECKPOINT PATH>
3. Evaluation: python main.py --mode evaluate --checkpoint < CHECKPOINT PATH > In case you want to dump the predictions just add: --output_file <OUTPUT_FILE_PATH>


[1] Cohen, A.A., Geva-Zatorsky, N., Eden, E., Frenkel-Morgenstern, M., Issaeva, I., Sigal, A., Milo, R., Cohen-Saidon, C., Liron, Y., Kam, Z., et al.: Dynamic proteomics of individual cancer cells in response to a drug. Science 322(5907), 1511–1516 (2008).
