{
  "targets": [{

      "target_name": "ncsdk",

      "sources": [
        "src/third_party/libusb/libusb/core.c",
        "src/third_party/libusb/libusb/descriptor.c",
        "src/third_party/libusb/libusb/hotplug.c",
        "src/third_party/libusb/libusb/io.c",
        "src/third_party/libusb/libusb/strerror.c",
        "src/third_party/libusb/libusb/sync.c",
        "src/mvnc_api.c",
        "src/usb_boot.c",
        "src/usb_link_vsc.c",
        "src/fp16.c",
        "src/init.cc"
      ],

      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "src/third_party/libusb/libusb",
      ],

      "cflags!" : [ "-fno-exceptions"],
      "cflags_cc!": [ "-fno-rtti",  "-fno-exceptions"],

      "conditions": [
        [ "OS==\"linux\"", {
            "cflags": [
              "-Wall"
            ]
        }],
        [ "OS==\"win\"", {
            "sources": [
              "src/third_party/libusb/libusb/os/poll_windows.c",
              "src/third_party/libusb/libusb/os/threads_windows.c",
              "src/third_party/libusb/libusb/os/windows_nt_common.c",
              "src/third_party/libusb/libusb/os/windows_winusb.c",
            ],
            "include_dirs": [
              "src/third_party/libusb/msvc",
            ],
            "cflags": [
              "-Wall"
            ],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "ExceptionHandling": "2",
                "DisableSpecificWarnings": [ "4530", "4506", "4244", "4200" ],
              },
            }
        }],
        [ # cflags on OS X are stupid and have to be defined like this
          "OS==\"mac\"", {
            "xcode_settings": {
              "OTHER_CFLAGS": [
                "-mmacosx-version-min=10.7",
                "-std=c++11",
                "-stdlib=libc++",
              ],
              "GCC_ENABLE_CPP_RTTI": "YES",
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
            }
          }
        ]
    ]
  }]
}
