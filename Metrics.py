from enum import Enum


class Metric(Enum):
    time = 1
    throughput = 2
    datasize = 3
    average_throughput = 4
    resource_utilization = 5
    wait_to_proc_time_ratio = 6
    #even_load_distribution_mse = 7 # load changed to queue load
    uncompressed_throughput = 8


'''
class MinimizeMetric(Metric):
    time = 1
    datasize = 2
    wait_to_proc_time_ratio = 3
    even_load_distribution = 4


class MaximizeMetric(Metric):
    throughput = 1
    average_throughput = 2
    resource_utilization = 3

'''

def get_metric(metric_p, result):
    """
    Returns the specified metric.

    Parameters:
        metric_p (enum or str): The metric to be calculated.
        result (dict): Dictionary containing all needed result metrics from a data transfer.

    Returns:
        float: Calculated metric-value.
    """

    #optimize for fastest transfer time
    if metric_p == Metric.time.name or metric_p == Metric.time:
        return float(result['time'])

    #maximize throughput, based on time and datasize
    elif metric_p == Metric.throughput.name or metric_p == Metric.throughput:
        return float(result['datasize']) / float(result['time'])

    #optimize for minimum network usage / transmitted data size
    elif metric_p == Metric.datasize.name or metric_p == Metric.datasize:
        return float(result['datasize'])

    #maximize average of all individual components throughputs
    elif metric_p == Metric.average_throughput.name or metric_p == Metric.average_throughput:

        total = (float(result['rcv_throughput']) +
                 float(result['decomp_throughput']) +
                 float(result['write_throughput']) +
                 float(result['read_throughput']) +
                 float(result['deser_throughput']) +
                 float(result['comp_throughput']) +
                 float(result['send_throughput']))

        average_throughput = total / 7
        return average_throughput

    #maxemize average compute ressource usage
    elif metric_p == Metric.resource_utilization.name or metric_p == Metric.resource_utilization:

        return -1

        #weighted_sum = (float(result['server_cpu']) * float(result['avg_cpu_server'])) + (float(result['client_cpu']) * float(result['avg_cpu_client']))
        #weighted_average = weighted_sum / (float(result['server_cpu']) + float(result['client_cpu']))
        #return weighted_sum

    #divide total wait time by total proc time, lower is better -> higher efficiency (?)
    elif metric_p == Metric.wait_to_proc_time_ratio.name or metric_p == Metric.wait_to_proc_time_ratio:
        return wait_to_proc_time_ratio(result)

    #average difference between individual load and average load. lower is better
    #elif metric_p == Metric.even_load_distribution_mse.name or metric_p == Metric.even_load_distribution_mse:
    #    return even_load_distribution_mse(result)

    elif metric_p == Metric.uncompressed_throughput.name or metric_p == Metric.uncompressed_throughput:
        if result['table'] == 'lineitem_sf10':
            uncompressed_throughput_value = (7715741636 / 1000 / 1000) / result['time']  # b to MB
            return uncompressed_throughput_value




def add_all_metrics_to_result(result):
    """
    Adds all available Metrics to a result-dictionary.

    Parameters:
        result (dict): Dictionary containing all needed result metrics from a data transfer.

    Returns:
        dict: Result-dict with added metrics.
    """

    for m in Metric:

        result_metric = get_metric(m,result)
        result[m.name] = result_metric

    return result



def wait_to_proc_time_ratio(result):
    """
    Calculates the ratio between the wait time and processing time for all components.
    Should be used as minimizing objective, because lower value indicates less wait time.

    Parameters:
        result (dict): Dictionary containing all needed result metrics from a data transfer.

    Returns:
        float: Ratio between wait time and processing time.
    """
    total_wait_time = (float(result['rcv_wait_time']) +
                       float(result['decomp_wait_time']) +
                       float(result['write_wait_time']) +
                       float(result['read_wait_time']) +
                       float(result['deser_wait_time']) +
                       float(result['comp_wait_time']) +
                       float(result['send_wait_time']))

    total_proc_time = (float(result['rcv_proc_time']) +
                       float(result['decomp_proc_time']) +
                       float(result['write_proc_time']) +
                       float(result['read_proc_time']) +
                       float(result['deser_proc_time']) +
                       float(result['comp_proc_time']) +
                       float(result['send_proc_time']))

    wait_to_proc_time_ration = total_wait_time / total_proc_time
    return wait_to_proc_time_ration


def even_load_distribution_mse(result):
    """
    Calculates the MSE of the different loads of all components.
    Should be used as minimizing objective, because lower value indicates more even load distribution.

    Parameters:
        result (dict): Dictionary containing all needed result metrics from a data transfer.

    Returns:
        float: MSE of loads.
    """
    total_load = (float(result['rcv_load']) +
                  float(result['decomp_load']) +
                  float(result['write_load']) +
                  float(result['read_load']) +
                  float(result['deser_load']) +
                  float(result['comp_load']) +
                  float(result['send_load']))

    average_load = total_load / 7

    total_difference_load = ((float(result['rcv_load']) - average_load)**2 +
                             (float(result['decomp_load']) - average_load)**2 +
                             (float(result['write_load']) - average_load)**2 +
                             (float(result['read_load']) - average_load)**2 +
                             (float(result['deser_load']) - average_load)**2 +
                             (float(result['comp_load']) - average_load)**2 +
                             (float(result['send_load']) - average_load)**2)

    return total_difference_load
