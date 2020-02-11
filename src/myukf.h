#include "Eigen/Dense"

class UKF{
    public:

    UKF();

    virtual ~UKF();

    void Init();

    void GenerateSigmaPoint(Eigen::MatrixXd* Xsig_out);
    void AugmentedSigmentPoints(Eigen::MatrixXd* Xsig_out);
    void SigmaPointPrediction(Eigen::MatrixXd* Xsig_out);
    void PredictMeanAndCovariance(Eigen::VectorXd* x_pred,Eigen::MatrixXd* p_pred);

    void PredictRadarMeasurement(Eigen::VectorXd* z_out,Eigen::MatrixXd* s_out);

    void UpdateState(Eigen::VectorXd* x_out,Eigen::MatrixXd* P_out);
};