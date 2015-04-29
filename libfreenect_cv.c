#include "libfreenect.h"
#include "libfreenect_sync.h"
#include "libfreenect_cv.h"


IplImage *freenect_sync_get_ir_cv(int index)
{
        static IplImage *image = 0;
        static char *data = 0;

        if (!image) {
                image = cvCreateImageHeader(cvSize(640,488), 8, 1);
        }
        unsigned int timestamp;

        if (freenect_sync_get_video((void**)&data, &timestamp, index, FREENECT_VIDEO_IR_8BIT)) {
                return NULL;
        }
        cvSetData(image, data, 640*1);
        return image;
}
