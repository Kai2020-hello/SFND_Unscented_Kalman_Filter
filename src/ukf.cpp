#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


/**
 * Initializes Unscented Kalman filter
 */

void NormalizeAngle(double& phi)
{
    phi = atan2(sin(phi), cos(phi));
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector 状态值
  x_ = VectorXd(5); 

  // initial covariance matrix 协方差 
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2 加速度噪音
  std_a_ = 30; 

  // Process noise standard deviation yaw acceleration in rad/s^2 角速度噪音
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m 激光雷达噪音横向
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m 激光雷达噪音纵向
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m  毫米波雷达噪音——？
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad 毫米波雷达噪音——？
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s 毫米波雷达噪音——？
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  // radar 维度
  n_z_ = 3;
  // 初始化 权重
  weights_ = VectorXd(2 * n_aug_+1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
      weights_(i) = 0.5/(lambda_ + n_aug_);
  }
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  //初始化
  if(!is_initialized_){
      if(meas_package.sensor_type_ == meas_package.LASER){
          //LASER
          const double x0 = meas_package.raw_measurements_[0];
          const double y0 = meas_package.raw_measurements_[0];
          x_ << x0, y0, 0, 0, 0;
      }else if (meas_package.sensor_type_ == meas_package.RADAR){
          // RADAR
          const double l = meas_package.raw_measurements_[0];
          const double alpha = meas_package.raw_measurements_[1];
          const double l1 = meas_package.raw_measurements_[2];

          x_ << l*cos(alpha),l*sin(alpha),l1,0,0;
      }
      if(fabs(x_(0)) < 0.001){
          x_(0) = 0.001;
      }
      if(fabs(x_(1)) < 0.001){
          x_(1) = 0.001;
      }
      P_ = MatrixXd::Identity(5,5);
      time_us_ = meas_package.timestamp_;
      return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  Prediction(delta_t);
  if(meas_package.sensor_type_ == meas_package.LASER){
    //LASER
      UKF::UpdateLidar(meas_package);
  }else if (meas_package.sensor_type_ == meas_package.RADAR){
    // RADAR
      UKF::UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // 生成sigma点
  MatrixXd Xsig = MatrixXd(n_aug_,2*n_aug_+1);
  AugmentedSigmentPoints(&Xsig);
  // 预测sigma点
  SigmaPointPrediction(Xsig,delta_t);
  // 根据sigma点预测 均值和协方差矩阵
  PredictMeanAndCovariance(&x_,&P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(2);
    MatrixXd S = MatrixXd(2, 2);
    // 计算预测测量值
    for (int i = 0; i < 2*n_aug_+1 ; ++i) {
        const double px = Xsig_pred_.col(i,0);
        const double py = Xsig_pred_.col(i,1);

        Zsig(i,0) = px;
        Zsig(i,0) = py;
    }
    z_pred = Zsig * weights_;

    // 预测协方差矩阵
    S.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1 ; ++i) {
        VectorXd diff = z_pred.col(i) - z_pred;
        NormalizeAngle(diff(1));

        S += weights_(i) * diff * diff.transpose();
    }

    S(0,0) += std_laspx_ * std_laspx_;
    S(1,1) += std_laspy_ * std_laspy_;

    UKF::UKF_Update(meas_package,Zsig,z_pred,S);
}

/**
 * 总的更新操作
 * @param meas_package
 * @param Zsig
 * @param z_pred
 * @param S
 */
void UKF::UKF_Update(MeasurementPackage meas_package, MatrixXd Zsig, VectorXd z_pred, MatrixXd S) {
    VectorXd z = meas_package.raw_measurements_;

    // 计算交叉关系
    MatrixXd T = MatrixXd(n_x,z.size());
    T.fill(0.0);
    for (int i = 0; i < 2*n_aug_ + 1; ++i) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        VectorXd z_diff = Zsig.col(i) - z_pred;

        NormalizeAngle(z_diff(1));
        NormalizeAngle(x_diff(3));

        T+= weights_(i) * x_diff * z_diff.transpose();
    }

    //计算增益
    MatrixXd K = T*S.inverse();

    VectorXd z_diff = z - z_pred;

    //angle normalization
    NormalizeAngle(z_diff(1));

    x_ += K*z_diff;
    P_ -= K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  //预测测量值 - 预测radar的值
  VectorXd z_pred = VectorXd(n_z_);
  MatrixXd s_pred = MatrixXd(n_z_,n_z_);
  MatrixXd Zsig = MatrixXd(n_z_,2 * n_aug_ + 1);
  PredictRadarMeasurement(&z_pred,&s_pred,&Zsig);

  // 相关性
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // 计算增益
  MatrixXd K = Tc*s_pred.inverse();

  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  //更新状态值
  x_ = x_ + K * z_diff;

  //更新 协方差矩阵
  P_ = P_ - K*s_pred*K.transpose();
}

/**
 * 生成扩增的sigma点
 */
void UKF::AugmentedSigmentPoints(MatrixXd* Xsig_out){
  // 扩展后的状态值
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // 扩展后的协方差矩阵
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  P_aug.fill(0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //计算扩展后的协方差矩阵的平方根
  MatrixXd A = P_aug.llt().matrixL();

  //计算sigma点
  MatrixXd Xsig = MatrixXd(n_aug_,2 * n_aug_ + 1);
  Xsig.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
  *Xsig_out = Xsig;

  std::cout << "Xsig = " << std::endl << Xsig << std::endl;
}

/**
 * 预测sigma点
 */
void UKF::SigmaPointPrediction(MatrixXd Xsig, double delta_t){
  
  for (int  i = 0; i < 2*n_aug_+1; i++)
  {
    double p_x = Xsig(0,i);
    double p_y = Xsig(1,i);
    double v = Xsig(2,i);
    double yaw = Xsig(3,i);
    double yawd = Xsig(4,i);
    double v_a = Xsig(5,i);
    double v_b = Xsig(6,i);

    // 位置
    double px_p,py_p;

    if (fabs(yaw) > 0.001)
    {// 防止除数为0
      px_p = p_x + v/yaw * (sin(yaw + yawd * delta_t) + sin(yaw));
      py_p = p_y + v/yaw * (cos(yaw) - cos(yaw + yawd * delta_t));
    }else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    // 速度，角速度，角速度加速度
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // 添加噪音
    px_p = px_p + 0.5 * v_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * v_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + delta_t*v_a;

    yaw_p = yaw_p + v_b * delta_t * delta_t;
    yawd_p = yawd_p + v_b * delta_t;
    
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
}

/**
 * 根据预测的sigma点预测均值和协方差
 */ 
void UKF::PredictMeanAndCovariance(VectorXd* x_pred,MatrixXd* p_pred){
    // 预测的均值和协方差
    VectorXd x = VectorXd(n_x_);
    MatrixXd P = MatrixXd(n_x_, n_x_);

    x.fill(0);
    P.fill(0);

    // 状态值结算
    for (int i = 0; i < 2*n_aug_+1; i++)
    {
      x += weights_(i)*Xsig_pred_.col(i);
    }
    // 协方差矩阵计算
    for (int i = 0; i < 2*n_aug_+1; i++)
    {
      VectorXd x_diff = Xsig_pred_.col(i) - x;

      while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

      P += weights_(i) * x_diff * x_diff.transpose(); 
    }

    *x_pred = x;
    *p_pred = P;
}

/**
 * 预测测量值 - 预测radar的值
 */
void UKF::PredictRadarMeasurement(Eigen::VectorXd* z_out,Eigen::MatrixXd* s_out,
                                  Eigen::MatrixXd* Zsig_out){
    // 存放 预测sigma点转化成测量空间点
    MatrixXd Zsig = MatrixXd(n_z_,2 * n_aug_+1); 
    
    // 预测的测量空间的均值
    VectorXd z_pred = VectorXd(n_z_);
    // 预测的测量空间的协方差
    MatrixXd S = MatrixXd(n_z_,n_z_);

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
      float p_x = Xsig_pred_(0,i);
      float p_y = Xsig_pred_(1,i);
      float v = Xsig_pred_(2,i);
      float yaw = Xsig_pred_(3,i);
      float yawd = Xsig_pred_(4,i);

      float th_2 = sqrt(p_x * p_x + p_y * p_y);

      float rho_z = th_2;
      float yaw_z = atan2(p_x,p_y);
      float rhod_z = (p_x * sin(yaw) * v + p_y * cos(yaw) * v ) / th_2;

      Zsig(0,i) = rho_z;
      Zsig(1,i) = yaw_z;
      Zsig(2,i) = rhod_z;
    }

    // 预测值赋值
    z_pred.fill(0);
    for (int i = 0; i < 2*n_aug_ + 1; i++)
    {
      z_pred += weights_(i) * Zsig.col(i);
    }
    // 对预测测量值的协方差矩阵 赋值
    S.fill(0);
    for (int i = 0; i < 2*n_aug_ + 1; i++)
    {
      VectorXd z_diff = Zsig.col(i) - z_pred;
      
      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //添加测量噪音 
    MatrixXd R = MatrixXd(n_z_,n_z_);
    R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;

    S += R;


    *z_out = z_pred;
    *s_out = S;
    *Zsig_out = Zsig;
}

