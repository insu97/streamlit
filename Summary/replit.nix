{ pkgs }: {
  deps = [
    pkgs.geckodriver
    (pkgs.chromium.override { version = "117.0.5938.88"; })
    (pkgs.chromedriver.override { version = "117.0.5938.88"; })
  ];
}
