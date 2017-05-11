{
  'targets': [
    {
      'target_name': 'tensorflow',
      'sources': [
        'src/tf.cpp',
        'src/tf_graph.cpp',
        'src/tf_session.cpp',
        'src/tf_tensor.cpp',
        'src/tf_datatype.cpp',
        'src/opencv.cpp',
      ],
      'include_dirs': [
        '<!(node -e "require(\'nan\')")',
        '<!(pwd)',
      ],
      'libraries': [
        '-L<!(pwd)/tensorflow',
        '-ltensorflow-lnx64',
        '-lopencv_core -lopencv_highgui -lopencv_imgproc',
        '-lm -lstdc++'
      ],
      'cflags_cc': [
        '-std=c++11',
        '-fexceptions',
        '-Wno-ignored-qualifiers'
      ],
      'ldflags': [
        '-Wl,-rpath,\$$ORIGIN/../lib'
      ]
    }
  ]
}
