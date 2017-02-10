{
  'targets': [
    {
      'target_name': 'caffe',
      'sources': [
        'js/caffejs.cpp',
      ],
      'include_dirs': [
        '<!(node -e "require(\'nan\')")',
        '<!(pwd)/src/',
        '<!(pwd)/include/',
        '<!(pwd)/__build/src/'
      ],
      'libraries': [
        '-L<!(pwd)/__build/lib',
        '-lcaffe',
        '-lopencv_core -lopencv_highgui -lopencv_imgproc',
        '-lm -lstdc++'
      ],
      'cflags_cc': [
        '-DCPU_ONLY=1',
        '-std=c++11',
        '-fexceptions',
        '-Wno-ignored-qualifiers'
      ]
    }
  ]
}
