#include "myukf.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

UKF::UKF(){
    Init();
}

UKF::~UKF(){

}

void UKF::Init(){

}

/**
 * 生成sigma点
 */
void UKF::GenerateSigmaPoint(MatrixXd* Xsig_out){
    //状态的维度
    int n_x = 5;
    //定义lambda
    int lambda = 3 - n_x;

    //定义状态量
    VectorXd x = VectorXd(n_x);
    x << 5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

    //定义协方差矩阵
    MatrixXd P = MatrixXd(n_x,n_x);
    P << 0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    //创建存储sigma点的容器
    MatrixXd Xsig = MatrixXd(n_x,2*n_x + 1);

    //计算P的平方根
    MatrixXd A = P.llt().matrixL();

    // 计算sigma点
    Xsig.col(0) = x;

    for (int i =0;i < n_x;i++){
        Xsig.col(i + 1) = x + sqrt(lambda + n_x)*A.col(i);
        Xsig.col(i + 1 + n_x) = x - sqrt(lambda + n_x)*A.col(i);
    }


    *Xsig_out = Xsig;

    std::cout << "Xsig = " << std::endl << Xsig << std::endl;
}

/**
 * 生成带有噪音的sigma点
 */
void UKF::AugmentedSigmentPoints(Eigen::MatrixXd* Xsig_out){


    //状态的维度
    int n_x = 5;

    //带有噪音的维度
    int n_aug = 7;


    //定义lambda
    int lambda = 3 - n_aug;

    //噪音-加速度
    double std_a = 0.2;

    //噪音-角速度噪音
    double std_yawdd = 0.2;

    //定义状态量
    VectorXd x = VectorXd(n_x);
    x << 5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

    //定义协方差矩阵
    MatrixXd P = MatrixXd(n_x,n_x);
    P << 0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    //扩展后的状态值
    VectorXd x_aug = VectorXd(n_aug);
    x_aug.head(5) = x;
    x_aug(5) = 0;
    x_aug(6) = 0;
    //扩展后的协方差矩阵
    MatrixXd P_aug = MatrixXd(n_aug,n_aug);
    P_aug.fill(0);
    P_aug.topLeftCorner(5,5) = P;
    P_aug(5,5) = std_a*std_a;
    P_aug(6,6) = std_yawdd*std_yawdd;

    //计算P_aug的平方根
    MatrixXd A = P_aug.llt().matrixL();

    //计算sigma点
    MatrixXd Xsig = MatrixXd(n_aug,2 * n_aug + 1);

    Xsig.col(0) = x_aug;

    for (int i = 0; i < n_aug; i++)
    {
        Xsig.col( i+1 ) = x_aug + sqrt(lambda + n_aug)*A.col(i);
        Xsig.col( i+1+n_aug ) = x_aug - sqrt(lambda + n_aug)*A.col(i);
    }
    *Xsig_out = Xsig;

    std::cout << "Xsig = " << std::endl << Xsig << std::endl;

}

/**
 * sigma点预测
 */
void UKF::SigmaPointPrediction(MatrixXd* Xsig_out){
    // 设置状态维度
    int n_x = 5;

    // 设置扩增后的维度
    int n_aug = 7;

    //sigma点
    MatrixXd Xsig_aug = MatrixXd(n_aug,2 * n_aug + 1);
    Xsig_aug << 
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
    1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

    //存放预测sigma点的容器
    MatrixXd Xsig_pred = MatrixXd(n_x,2 * n_aug + 1);

    // 间隔时间
    double delta_t = 0.1;


    // 预测sigma点
    for (int i = 0; i < 2*n_aug + 1; i++)
    {
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double v_a = Xsig_aug(5,i);
        double v_b = Xsig_aug(6,i);

        double px_p,py_p;

        //避免除数为0
        if(fabs(yaw) > 0.001){
            px_p = p_x + v/yaw *(sin(yaw + yawd*delta_t)-sin(yaw));
            px_p = p_y + v/yaw *(cos(yaw) - cos(yaw + yawd*delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            px_p = p_y + v * delta_t * sin(yaw);
        }

        //速度
        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //添加噪音
        px_p = px_p + 0.5 * v_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * v_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + v_a * delta_t;

        yaw_p = yaw_p + 0.5 * v_b * delta_t * delta_t;
        yawd_p = yawd_p + v_b * delta_t;

        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }

    std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

    *Xsig_out = Xsig_pred;
}

/**
 * 根据预测的sigma点预测均值和协方差
 */ 
void UKF::PredictMeanAndCovariance(Eigen::VectorXd* x_pred,Eigen::MatrixXd* p_pred){
    int n_x = 5;

    int n_aug = 7;

    double lambda = 3 - n_x;

    //预测的sigma点
    MatrixXd Xsig_pred = MatrixXd(n_x,2*n_aug+1);
    Xsig_pred << 
        5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
        1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
        2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
        0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
        0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

    //sigma点权重
    VectorXd weights = VectorXd(2*n_aug+1);

    weights(0) = lambda/(lambda + n_aug);

    for (int i = 1; i < 2*n_aug + 1; i++)
    {
        weights(i) = 0.5/(lambda + n_aug);
    }
    

    //预测的均值，协方差
    VectorXd x = VectorXd(n_x);
    MatrixXd P = MatrixXd(n_x, n_x);

    x.fill(0);
    P.fill(0);

    //x状态值计算
    for (int i = 0; i < 2*n_aug+1; i++)
    {
        x = x + weights(i)* Xsig_pred.col(i);
    }

    //协方差矩阵预测
    for (size_t i = 0; i < 2*n_aug+1; i++)
    {
        VectorXd x_diff = Xsig_pred.col(i) - x;

        while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;
        
        P = P + weights(i)*x_diff*x_diff.transpose();
    }
    
    *x_pred = x;
    *p_pred = P;

    std::cout << "Predicted state" << std::endl;
    std::cout << x << std::endl;
    std::cout << "Predicted covariance matrix" << std::endl;
    std::cout << P << std::endl;

}


void UKF::PredictRadarMeasurement(VectorXd* z_out,MatrixXd* s_out){
    int n_x = 5;

    int n_aug = 7;

    // 测量空间维度
    int n_z = 3;

    double lambda = 3 - n_aug;

    //sigma点权重
    VectorXd weights = VectorXd(2*n_aug+1);
    weights(0) = lambda/(lambda + n_aug);
    for (int i = 1; i < 2*n_aug + 1; i++)
    {
        weights(i) = 0.5/(lambda + n_aug);
    }
    //radar噪音
    double std_radr = 0.3;
    double std_radphi = 0.0175;
    double std_radrd = 0.1;

    //预测的sigma点
    MatrixXd Xsig_pred = MatrixXd(n_x,2 * n_aug + 1);

    Xsig_pred <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
    // 存放 预测sigma点转化成测量空间点
    MatrixXd Zsig = MatrixXd(n_z,2*n_aug+1);
    
    // 预测的测量空间的均值
    VectorXd z_pred = VectorXd(n_z);
    // 预测的测量空间的协方差
    MatrixXd S = MatrixXd(n_z,n_z);

    for (int i = 0; i < 2 * n_aug + 1; i++)
    {
        float p_x = Xsig_pred(0,i);
        float p_y = Xsig_pred(1,i);
        float v = Xsig_pred(2,i);
        float yaw = Xsig_pred(3,i);
        float yawd = Xsig_pred(4,i);


        float th_2 = sqrt(p_x*p_x + p_y*p_y);

        float rho_z = th_2;
        float yaw_z = atan2(p_y,p_x);
        float rhod_z = (p_x*cos(yaw)*v + p_y*sin(yaw)*v)/th_2;

        Zsig(0,i) = rho_z;
        Zsig(1,i) = yaw_z;
        Zsig(2,i) = rhod_z;
    }

    z_pred.fill(0.0);

    for (int i = 0; i < 2*n_aug + 1; i++)
    {
        z_pred += weights(i)*Zsig.col(i);
    }

    //测量空间的协方差矩阵
    S.fill(0.0);
    for (int i = 0; i < 2*n_aug; i++)
    {
        VectorXd z_diff = Zsig.col(i) - z_pred;
      
        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    //添加测量噪音 
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<  std_radr*std_radr, 0, 0,
        0, std_radphi*std_radphi, 0,
        0, 0,std_radrd*std_radrd;

    S += R;

    *z_out = z_pred;
    *s_out = S;

    std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    std::cout << "S: " << std::endl << S << std::endl;
}

void UKF::UpdateState(VectorXd* x_out,MatrixXd* P_out){
    int n_x = 5;

    int n_aug = 7;

    int n_z = 3;

    double lambda = 3 - n_aug;

    //sigma点权重
    VectorXd weights = VectorXd(2*n_aug+1);
    weights(0) = lambda/(lambda + n_aug);
    for (int i = 1; i < 2*n_aug + 1; i++)
    {
        weights(i) = 0.5/(lambda + n_aug);
    }

    // 预测的sigma点
    MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
    Xsig_pred <<
     5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
       1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
      2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
     0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
      0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

    // 预测的均值
    VectorXd x = VectorXd(n_x);
    x <<
    5.93637,
    1.49035,
    2.20528,
    0.536853,
    0.353577;

    // 预测 协方差
    MatrixXd P = MatrixXd(n_x,n_x);
    P <<
        0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
        -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
        0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
        -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
        -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;

    // 预测的测量空间值
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
    Zsig <<
    6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
    0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
    2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

    // 预测的测量空间的均值
    VectorXd z_pred = VectorXd(n_z);
    z_pred <<
        6.12155,
        0.245993,
        2.10313;

    // 预测的测量空间的协方差
    MatrixXd S = MatrixXd(n_z,n_z);
    S <<
        0.0946171, -0.000139448,   0.00407016,
        -0.000139448,  0.000617548, -0.000770652,
        0.00407016, -0.000770652,    0.0180917;

    // 测量值
    VectorXd z = VectorXd(n_z);
    z <<
        5.9214,   // rho in m
        0.2187,   // phi in rad
        2.0062;   // rho_dot in m/s

    // 相关性
    MatrixXd Tc = MatrixXd(n_x, n_z);

    for (int  i = 0; i < 2 * n_aug + 1; i++)
    {
        VectorXd x_diff = Xsig_pred.col(i) - x;
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    // 计算增益
    MatrixXd K = Tc*S.inverse();

    // 计算预测值和测量值的差值
    VectorXd z_diff = z - z_pred;

    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //更新状态值
    x = x + K * z_diff;

    //更新 协方差矩阵
    P = P - K*S*K.transpose();

    std::cout << "Updated state x: " << std::endl << x << std::endl;
    std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

    *x_out = x;
    *P_out = P;
}



int main(){
    UKF ukf;

    // MatrixXd Xsig = MatrixXd(5,15);
    // VectorXd x = VectorXd(5);
    // MatrixXd P = MatrixXd(5,5);
    // ukf.PredictMeanAndCovariance(&x,&P);

    // VectorXd z = VectorXd(3);
    // MatrixXd S = MatrixXd(3,3);

    // ukf.PredictRadarMeasurement(&z,&S);

    VectorXd x = VectorXd(5);
    MatrixXd P = MatrixXd(5,5);

    ukf.UpdateState(&x,&P);

}