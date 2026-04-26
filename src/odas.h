struct OdasParams {
    int   lanczosQ        = 3;

    float cannyLowThresh  = 50.0f;
    float cannyHighThresh = 150.0f;

    // K=51.0f because paper uses [0,1] normalized images.
    // We work in [0,255], so K_ours = K_paper * 255 = 0.2 * 255 = 51.0
    // With K=0.2 on [0,255] data: exp(-(g/0.2)^2) = 0 for g > 0.5
    // → effectively kills ALL diffusion → no visible effect
    float K               = 51.0f;   // WAS 0.20f — THIS WAS THE ROOT CAUSE

    int   odadIterations  = 1;

    // CS params reused as GSS iterations (20-50 sufficient for GSS)
    int   csNests         = 25;
    int   csMaxIter       = 100;
    float csPa            = 0.25f;
    float csLevyBeta      = 1.5f;

    bool  useYCbCr        = true;
};
