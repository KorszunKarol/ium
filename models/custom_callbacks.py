import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import mean_absolute_error, r2_score
import io

matplotlib.use("Agg")


def asymmetric_weighted_male_loss(w_u=1.5, w_o=1.0):
    """
  Create Asymmetric Weighted MALE (Mean Absolute Log Error) loss function.

  Args:
      w_u (float): Weight for under-predictions (y_pred < y_true). Default 1.5
      w_o (float): Weight for over-predictions (y_pred > y_true). Default 1.0

  Returns:
      TensorFlow loss function
  """

    def loss_fn(y_true, y_pred):

        epsilon = tf.keras.backend.epsilon()


        y_true_safe = tf.maximum(y_true, epsilon)
        y_pred_safe = tf.maximum(y_pred, epsilon)


        log_true = tf.math.log1p(y_true_safe)
        log_pred = tf.math.log1p(y_pred_safe)


        abs_log_error = tf.abs(log_true - log_pred)



        under_prediction_mask = tf.cast(log_pred < log_true, tf.float32)
        over_prediction_mask = tf.cast(log_pred >= log_true, tf.float32)


        weighted_error = (under_prediction_mask * w_u +
                          over_prediction_mask * w_o) * abs_log_error

        return tf.reduce_mean(weighted_error)

    return loss_fn


def quantile_loss(quantile=0.6):
    """
  Create Quantile Loss function for targeting specific percentiles.

  Args:
      quantile (float): Target quantile (e.g., 0.6 for 60th percentile)

  Returns:
      TensorFlow loss function
  """

    def loss_fn(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(quantile * error, (quantile - 1) * error))

    return loss_fn


class CustomMetricsCallback(callbacks.Callback):

    def __init__(
        self,
        log_dir,
        X_val,
        y_val_log,
        y_val_original_scale,
        original_features_val=None,
        plape_thresholds_pct=[10, 20, 30],
        enable_bias_plots=True,
        enable_nn_internal_plots=True,
        enable_distributional_plots=True,
        plot_frequencies=None,
        max_log_value=None,
        max_revenue_value=1000000,
        asymmetric_weights=(1.5, 1.0)):
        super().__init__()
        self.log_dir = log_dir
        self.X_val = X_val
        self.y_val_log = y_val_log
        self.y_val_original_scale = y_val_original_scale
        self.original_features_val = original_features_val
        self.writer = tf.summary.create_file_writer(log_dir)
        self.plot_every_n_epochs = 10
        self.enable_custom_plots = original_features_val is not None
        self.plape_thresholds_pct = plape_thresholds_pct

        self.enable_bias_plots = enable_bias_plots
        self.enable_nn_internal_plots = enable_nn_internal_plots
        self.enable_distributional_plots = enable_distributional_plots

        self.max_log_value = max_log_value
        self.max_revenue_value = max_revenue_value


        self.w_u, self.w_o = asymmetric_weights

        default_frequencies = {
            'bias_analysis': 5,
            'distributional': 10,
            'nn_internals': 20,
            'original_plots': 10,
        }
        self.plot_frequencies = plot_frequencies if plot_frequencies else default_frequencies

    def _calculate_male(self, y_true, y_pred):
        """Calculate Mean Absolute Log Error (MALE)"""
        try:

            log_true = np.log1p(np.maximum(y_true, 0))
            log_pred = np.log1p(np.maximum(y_pred, 0))
            male = np.mean(np.abs(log_true - log_pred))
            return male
        except Exception as e:
            print(f"Error calculating MALE: {e}")
            return np.nan

    def _calculate_asymmetric_male(self, y_true, y_pred, w_u=None, w_o=None):
        """Calculate Asymmetric Weighted MALE"""
        try:
            if w_u is None:
                w_u = self.w_u
            if w_o is None:
                w_o = self.w_o


            log_true = np.log1p(np.maximum(y_true, 0))
            log_pred = np.log1p(np.maximum(y_pred, 0))


            abs_log_error = np.abs(log_true - log_pred)


            under_prediction_mask = (log_pred < log_true).astype(np.float32)
            over_prediction_mask = (log_pred >= log_true).astype(np.float32)


            weighted_error = (under_prediction_mask * w_u +
                              over_prediction_mask * w_o) * abs_log_error

            return np.mean(weighted_error)
        except Exception as e:
            print(f"Error calculating Asymmetric MALE: {e}")
            return np.nan

    def _calculate_wape(self, y_true, y_pred):
        """Calculate Weighted Absolute Percentage Error (WAPE)."""
        try:

            y_true_np = np.array(y_true)
            y_pred_np = np.array(y_pred)

            sum_abs_actual = np.sum(np.abs(y_true_np))
            if sum_abs_actual == 0:
                return np.nan

            sum_abs_error = np.sum(np.abs(y_true_np - y_pred_np))
            wape = (sum_abs_error / sum_abs_actual) * 100.0
            return wape
        except Exception as e:
            print(f"Error calculating WAPE: {e}")
            return np.nan

    def _calculate_prediction_bias_metrics(self, y_true, y_pred):
        """Calculate detailed bias metrics"""
        try:

            bias = np.mean(y_pred - y_true)


            under_predictions = np.mean(y_pred < y_true) * 100


            median_bias = np.median(y_pred - y_true)


            log_true = np.log1p(np.maximum(y_true, 0))
            log_pred = np.log1p(np.maximum(y_pred, 0))
            log_bias = np.mean(log_pred - log_true)

            return {
                'bias': bias,
                'under_predictions_pct': under_predictions,
                'median_bias': median_bias,
                'log_bias': log_bias
            }
        except Exception as e:
            print(f"Error calculating bias metrics: {e}")
            return {}

    def _plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _plot_predictions_vs_actual(self, y_true, y_pred, title):
        """Create scatter plot of predictions vs actual values"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Actual Revenue ($)')
            ax.set_ylabel('Predicted Revenue ($)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            return fig
        except Exception as e:
            print(f"Error in _plot_predictions_vs_actual: {e}")
            return None

    def _plot_residuals_histogram(self, residuals, title):
        """Create histogram of residuals"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='Zero residual')
            ax.set_xlabel('Residuals ($)')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            return fig
        except Exception as e:
            print(f"Error in _plot_residuals_histogram: {e}")
            return None

    def _log_bias_by_segments(self, y_true, y_pred, epoch):
        """Log bias analysis by revenue segments"""
        try:
            percentiles = [0, 25, 50, 75, 90, 100]
            segments = np.percentile(y_true, percentiles)

            for i in range(len(segments) - 1):
                mask = (y_true >= segments[i]) & (y_true < segments[i + 1])
                if np.sum(mask) > 0:
                    segment_bias = np.mean(y_pred[mask] - y_true[mask])
                    tf.summary.scalar(f"val_bias_segment_{i+1}",
                                      segment_bias,
                                      step=epoch)
        except Exception as e:
            print(f"Error in _log_bias_by_segments: {e}")

    def _log_plape_metrics(self, y_true_orig, y_pred_orig, epoch):
        """Log Percentage of Absolute Percentage Error (PLAPE) within defined thresholds."""
        try:
            if not self.plape_thresholds_pct:
                return

            absolute_errors = np.abs(y_true_orig - y_pred_orig)
            relative_errors = absolute_errors / np.maximum(y_true_orig, 1.0)

            for threshold_pct in self.plape_thresholds_pct:
                threshold_decimal = threshold_pct / 100.0
                metric_value = np.mean(
                    relative_errors <= threshold_decimal) * 100
                tf.summary.scalar(f"val_plape_{threshold_pct}pct",
                                  metric_value,
                                  step=epoch)
        except Exception as e:
            print(f"Error in _log_plape_metrics: {e}")

    def _plot_comprehensive_error_analysis(self, y_true, y_pred):
        """Create comprehensive error analysis plot"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                         2,
                                                         figsize=(15, 12))

            errors = np.abs(y_true - y_pred)
            relative_errors = errors / np.maximum(y_true, 1) * 100


            ax1.scatter(y_true, errors, alpha=0.5, s=20)
            ax1.set_xlabel('Actual Revenue ($)')
            ax1.set_ylabel('Absolute Error ($)')
            ax1.set_title('Prediction Error vs Actual Value')
            ax1.grid(True, alpha=0.3)


            ax2.hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(np.median(relative_errors),
                        color='red',
                        linestyle='--',
                        label=f'Median: {np.median(relative_errors):.1f}%')
            ax2.set_xlabel('Relative Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Relative Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)


            n_bins = 10
            bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
            bin_errors = []

            for i in range(len(bin_edges) - 1):
                mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    bin_errors.append(np.mean(errors[mask]))
                else:
                    bin_errors.append(0)

            ax3.bar(range(len(bin_errors)), bin_errors, alpha=0.7)
            ax3.set_xlabel('Prediction Bins (Low to High)')
            ax3.set_ylabel('Mean Absolute Error ($)')
            ax3.set_title('Error by Prediction Range')
            ax3.grid(True, alpha=0.3)


            sorted_errors = np.sort(relative_errors)
            cumulative = np.arange(
                1,
                len(sorted_errors) + 1) / len(sorted_errors) * 100

            ax4.plot(sorted_errors, cumulative, linewidth=2)
            ax4.axvline(10, color='green', linestyle='--', label='10%')
            ax4.axvline(25, color='orange', linestyle='--', label='25%')
            ax4.axvline(50, color='red', linestyle='--', label='50%')
            ax4.set_xlabel('Relative Error (%)')
            ax4.set_ylabel('Cumulative Percentage')
            ax4.set_title('Cumulative Error Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error in _plot_comprehensive_error_analysis: {e}")
            return None

    def _create_performance_dashboard(self, y_true, y_pred, epoch):
        """Create performance dashboard with key metrics"""
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)


            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            r2 = r2_score(y_true, y_pred)
            male = self._calculate_male(y_true, y_pred)
            wape = self._calculate_wape(y_true, y_pred)

            errors = np.abs(y_true - y_pred)
            relative_errors = errors / np.maximum(y_true, 1) * 100

            within_10 = np.mean(relative_errors <= 10) * 100
            within_25 = np.mean(relative_errors <= 25) * 100
            within_50 = np.mean(relative_errors <= 50) * 100


            ax_metrics = fig.add_subplot(gs[0, :])
            ax_metrics.axis('off')

            metrics_text = f"Epoch {epoch} | MAE: ${mae:.0f} | RMSE: ${rmse:.0f} | R²: {r2:.3f} | MALE: {male:.3f} | WAPE: {wape:.1f}%"
            ax_metrics.text(0.05,
                            0.5,
                            metrics_text,
                            transform=ax_metrics.transAxes,
                            fontsize=14,
                            fontweight='bold')

            accuracy_text = f"Within 10%: {within_10:.1f}% | Within 25%: {within_25:.1f}% | Within 50%: {within_50:.1f}%"
            ax_metrics.text(0.05,
                            0.2,
                            accuracy_text,
                            transform=ax_metrics.transAxes,
                            fontsize=12)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error in _create_performance_dashboard: {e}")
            return None

    def _create_bias_analysis_plot(self, y_true, y_pred):
        """Create bias analysis plot with bias vs. actual and bias by segments."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            bias = y_pred - y_true


            ax1.scatter(y_true, bias, alpha=0.5, s=15, label='Bias points')
            ax1.axhline(0,
                        color='red',
                        linestyle='--',
                        lw=2,
                        label='Zero Bias')


            if len(y_true) > 1 and len(bias) > 1:
                try:
                    z = np.polyfit(y_true, bias, 1)
                    p = np.poly1d(z)
                    ax1.plot(np.sort(y_true),
                             p(np.sort(y_true)),
                             "orange",
                             linestyle='-.',
                             lw=2,
                             label='Trend')
                except np.linalg.LinAlgError:
                    print(
                        "Could not fit trend line for bias vs actual: LinAlgError"
                    )
                except Exception as e_trend:
                    print(
                        f"Error fitting trend line for bias vs actual: {e_trend}"
                    )

            ax1.set_xlabel('Actual Revenue ($)')
            ax1.set_ylabel('Bias (Predicted - Actual) ($)')
            ax1.set_title('Prediction Bias vs. Actual Revenue')
            ax1.legend()
            ax1.grid(True, alpha=0.3)



            y_true_1d = np.ravel(y_true)


            try:
                percentiles = [0, 20, 40, 60, 80, 100]
                segment_edges = np.percentile(y_true_1d, percentiles)

                segment_edges = np.unique(segment_edges)

                if len(segment_edges
                       ) <= 1:
                    print(
                        "Not enough unique segment edges to create bias by segments plot. Skipping."
                    )

                    ax2.text(0.5,
                             0.5,
                             "Insufficient data for segmented bias plot",
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=ax2.transAxes)
                else:
                    segment_data = []
                    segment_labels = []
                    for i in range(len(segment_edges) - 1):
                        low_edge = segment_edges[i]
                        high_edge = segment_edges[i + 1]

                        if i == len(segment_edges) - 2 and percentiles[
                                i + 1] == 100:
                            mask = (y_true_1d >= low_edge) & (y_true_1d
                                                              <= high_edge)
                        else:
                            mask = (y_true_1d >= low_edge) & (y_true_1d
                                                              < high_edge)

                        if np.sum(mask) > 0:
                            segment_data.append(bias[mask])
                            segment_labels.append(
                                f"${low_edge:,.0f} - ${high_edge:,.0f}")
                        elif i == 0 and low_edge == high_edge:
                            mask = (y_true_1d == low_edge)
                            if np.sum(mask) > 0:
                                segment_data.append(bias[mask])
                                segment_labels.append(f"${low_edge:,.0f}")

                    if segment_data:
                        ax2.boxplot(segment_data,
                                    labels=segment_labels,
                                    patch_artist=True,
                                    medianprops={'color': 'red'})
                        ax2.axhline(0, color='red', linestyle='--', lw=2)
                        ax2.set_xlabel('Actual Revenue Segments')
                        ax2.set_ylabel('Bias (Predicted - Actual) ($)')
                        ax2.set_title(
                            'Prediction Bias Distribution by Revenue Segments')
                        ax2.tick_params(axis='x', rotation=30)
                        ax2.grid(True, axis='y', alpha=0.3)
                    else:
                        ax2.text(0.5,
                                 0.5,
                                 "No data points in defined segments.",
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=ax2.transAxes)
            except Exception as e_segment:
                print(f"Error creating bias by segments plot: {e_segment}")
                ax2.text(0.5,
                         0.5,
                         f"Error: {e_segment}",
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax2.transAxes)

            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in _create_bias_analysis_plot: {e}")
            return None

    def on_epoch_end(self, epoch, logs=None):

        y_pred_tensor = self.model.predict(self.X_val, verbose=0)
        y_pred_numpy = y_pred_tensor.flatten()


        if self.max_revenue_value is not None:
            y_pred_numpy = np.clip(y_pred_numpy, a_min=0, a_max=self.max_revenue_value)


        y_pred_original_scale = y_pred_numpy

        with self.writer.as_default():

            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            tf.summary.scalar("learning_rate", lr, step=epoch)


            mae_orig_metric = tf.keras.metrics.MeanAbsoluteError()
            mae_orig_metric.update_state(self.y_val_original_scale, y_pred_original_scale)
            tf.summary.scalar("val_mae_original_scale", mae_orig_metric.result(), step=epoch)

            mse_orig_metric = tf.keras.metrics.MeanSquaredError()
            mse_orig_metric.update_state(self.y_val_original_scale, y_pred_original_scale)
            tf.summary.scalar("val_rmse_original_scale", tf.sqrt(mse_orig_metric.result()), step=epoch)

            r2_orig = r2_score(self.y_val_original_scale.reshape(-1), y_pred_original_scale.reshape(-1))
            tf.summary.scalar("val_r2_original_scale", r2_orig, step=epoch)


            male_metric = self._calculate_male(self.y_val_original_scale, y_pred_original_scale)
            wape_metric = self._calculate_wape(self.y_val_original_scale, y_pred_original_scale)


            asymmetric_male_metric = self._calculate_asymmetric_male(
                self.y_val_original_scale, y_pred_original_scale)

            tf.summary.scalar("val_male_original_scale", male_metric, step=epoch)
            tf.summary.scalar("val_wape_original_scale", wape_metric, step=epoch)
            tf.summary.scalar("val_asymmetric_male_original_scale", asymmetric_male_metric, step=epoch)


            bias_metrics = self._calculate_prediction_bias_metrics(
                self.y_val_original_scale, y_pred_original_scale)

            for metric_name, metric_value in bias_metrics.items():
                tf.summary.scalar(f"val_{metric_name}",
                                  metric_value,
                                  step=epoch)


            abs_errors = np.abs(self.y_val_original_scale -
                                y_pred_original_scale)
            relative_errors = abs_errors / np.maximum(
                self.y_val_original_scale, 1)

            within_10pct = np.mean(relative_errors <= 0.1) * 100
            within_25pct = np.mean(relative_errors <= 0.25) * 100
            within_50pct = np.mean(relative_errors <= 0.5) * 100

            tf.summary.scalar("val_predictions_within_10pct",
                              within_10pct,
                              step=epoch)
            tf.summary.scalar("val_predictions_within_25pct",
                              within_25pct,
                              step=epoch)
            tf.summary.scalar("val_predictions_within_50pct",
                              within_50pct,
                              step=epoch)


            self._log_bias_by_segments(self.y_val_original_scale,
                                       y_pred_original_scale, epoch)
            self._log_plape_metrics(self.y_val_original_scale,
                                    y_pred_original_scale, epoch)


            scatter_plot_fig = self._plot_predictions_vs_actual(
                self.y_val_original_scale, y_pred_original_scale,
                "Predictions vs Actual (Original Scale Validation)")
            if scatter_plot_fig:
                tf.summary.image(
                    "Predictions vs Actual (Original Scale Validation)",
                    self._plot_to_image(scatter_plot_fig),
                    step=epoch)

            residuals_original_scale = self.y_val_original_scale - y_pred_original_scale
            residuals_hist_fig = self._plot_residuals_histogram(
                residuals_original_scale,
                "Residuals Distribution (Original Scale Validation)")
            if residuals_hist_fig:
                tf.summary.image(
                    "Residuals Distribution (Original Scale Validation)",
                    self._plot_to_image(residuals_hist_fig),
                    step=epoch)


            tf.summary.histogram("actual_values_val_original_scale",
                                 self.y_val_original_scale,
                                 step=epoch)
            tf.summary.histogram("predicted_values_val_original_scale",
                                 y_pred_original_scale,
                                 step=epoch)
            tf.summary.histogram("prediction_errors_val",
                                 abs_errors,
                                 step=epoch)


            if epoch % self.plot_every_n_epochs == 0 and self.enable_custom_plots:
                comprehensive_error_fig = self._plot_comprehensive_error_analysis(
                    self.y_val_original_scale, y_pred_original_scale)
                if comprehensive_error_fig:
                    tf.summary.image(
                        "Comprehensive Error Analysis",
                        self._plot_to_image(comprehensive_error_fig),
                        step=epoch)

                dashboard_fig = self._create_performance_dashboard(
                    self.y_val_original_scale, y_pred_original_scale, epoch)
                if dashboard_fig:
                    tf.summary.image("Performance Dashboard",
                                     self._plot_to_image(dashboard_fig),
                                     step=epoch)


                bias_analysis_fig = self._create_bias_analysis_plot(
                    self.y_val_original_scale, y_pred_original_scale)
                if bias_analysis_fig:
                    tf.summary.image("Bias Analysis",
                                     self._plot_to_image(bias_analysis_fig),
                                     step=epoch)

            self.writer.flush()
