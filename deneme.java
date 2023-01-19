public class deneme {
    public static void main(String[] args) {
        double[] w = {1,2,3};
        double b = 8;
        double[][] x = {{5, 12, 13}, {3, 4, 5}, {1, 2, 3}, {2, 3, 1}};
        double[] y = {179, 77, 47, 41};
       // double learningRate = 0.018;
       double learningRate = 0.03;
        LinearRegression linearReg = new LinearRegression(w, b, x, y, learningRate);
        linearReg.doLinearRegression();
    }
}
