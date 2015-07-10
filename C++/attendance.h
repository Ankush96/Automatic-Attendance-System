#ifndef __ATTENDANCE_H_INCLUDED__
#define __ATTENDANCE_H_INCLUDED__

std::string prediction_name(int);

void tune_seg_params(std::string, int, int, int, int, int);

int video_recognizer(int, int, int, int);

void model_main( std::string, int, bool, int, int, int, int);

void image_recognizer(std::string, int, int, int, int, int, int, int);

#endif
