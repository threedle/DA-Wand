import os
import time

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None

class Writer:
    def __init__(self, opt, natural=False):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.nexamples = 0
        self.ncorrect = 0
        self.natural = natural 
        self.start_logs()
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            if self.opt.continue_train == True and os.path.exists(self.log_name):
                with open(self.log_name, "a") as log_file:
                    now = time.strftime("%c")
                    log_file.write('================ Training Loss (%s) ================\n' % now)
            else:
                with open(self.log_name, "w") as log_file:
                    now = time.strftime("%c")
                    log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                if self.natural:
                    log_file.write('--------- Natural Test (%s) ---------\n' % now)
                else:
                    log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data, loss_breakdown=None, distortion=None):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses)
        if distortion: 
            message = '([Self Supervision] epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f, distortion: %.3f' \
                  % (epoch, i, t, t_data, losses, distortion)
        # Print loss breakdown as well if given 
        if loss_breakdown is not None: 
            message += "\n\t [Loss Breakdown]"
            for name, value in loss_breakdown.items():
                message += f" {name}: {value:0.4f}, "
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_stats(self, epoch, avgstats, statnames, distortion=False):
        """ prints test accuracy to terminal / file """
        if distortion: 
            message = f"\t[Distortion] {epoch}"
        else:
            message = f"epoch: {epoch}"
        for i in range(len(avgstats)):
            val = avgstats[i]
            key = statnames[i]
            if val // 1e5 >= 1: 
                message += f", TEST {key}: [{val:.3e}]"
            else:
                message += f", TEST {key}: [{val:.3f}]"
        message += "\n"
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
