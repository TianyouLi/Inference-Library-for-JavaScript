{
  'targets': [
    {
      'target_name': 'caffe',
      'sources': [
        'js/caffejs.cpp',
        'js/caffejs_symbols.cpp'
      ],
      'include_dirs': [
        '<!(node -e "require(\'nan\')")',
        '<!(pwd)/src/',
        '<!(pwd)/include/',
        '<!(pwd)/build/src/'
      ],
      'libraries': [
        '-L<!(pwd)/build/lib',
        '<!(pwd)/build/lib/libcaffe.a',
	'<!(if [ -d /usr/local/cuda/lib ]; then echo "-L/usr/local/cuda/lib"; fi)',
        '<!(if [ -d /usr/local/cuda/lib64 ]; then echo "-L/usr/local/cuda/lib64"; fi)',
        '<!(if [ -d /usr/local/cuda ]; then echo "-lcudart"; fi)',
	'-lcudart -lcublas -lcurand -lcudnn',	
        '-L/home/kanghua/mklml_lnx_2017.0.2.20170110/lib -lmklml_gnu',
	#'-L/opt/intel/lib/intel64_lin -L/opt/intel/mkl/lib/intel64 -lmkl_rt',
        '-lglog -lprotobuf',
        '-lboost_system -lboost_filesystem -lboost_regex -lboost_thread',
        '-lhdf5_hl -lhdf5',
        '-lopencv_core -lopencv_highgui -lopencv_imgproc',
        '-lm -lstdc++'
      ],
      'ldflags': [
	'<!(if [ -d /usr/local/cuda/lib ]; then echo "-Wl,-rpath,/usr/local/cuda/lib"; fi)',
        '<!(if [ -d /usr/local/cuda/lib64 ]; then echo "-Wl,-rpath,/usr/local/cuda/lib64"; fi)',
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
