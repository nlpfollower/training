load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# rules_oci
http_archive(
    name = "rules_oci",
    sha256 = "e96d70faa4bace3e09fdb1d7d1441b838920f491588889ff9a7e2615afca5799",
    strip_prefix = "rules_oci-2.0.0-alpha2",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v2.0.0-alpha2/rules_oci-v2.0.0-alpha2.tar.gz",
)

load("@rules_oci//oci:dependencies.bzl", "rules_oci_dependencies")
rules_oci_dependencies()

load("@rules_oci//oci:repositories.bzl", "oci_register_toolchains")
oci_register_toolchains(name = "oci")

# Pull the base OCI image
load("@rules_oci//oci:pull.bzl", "oci_pull")
oci_pull(
    name = "base_image",
    registry = "index.docker.io",
    repository = "continuumio/miniconda3",
    tag = "latest",
    platforms = ["linux/amd64"],
)

# rules_pkg
http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
    ],
    sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()