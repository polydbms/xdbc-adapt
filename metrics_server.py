
class MetricsServer:
    def __init__(self, transfer_id, total_time, read_wait_time, read_proc_time, read_throughput, read_throughput_pb,
                 free_load,
                 deser_wait_time, deser_proc_time, deser_throughput, deser_throughput_pb, deser_load,
                 comp_wait_time, comp_proc_time, comp_throughput, comp_throughput_pb, comp_load,
                 send_wait_time, send_proc_time, send_throughput, send_throughput_pb, send_load):
        self.transfer_id = transfer_id
        self.total_time = total_time
        self.read_wait_time = read_wait_time
        self.read_proc_time = read_proc_time
        self.read_throughput = read_throughput
        self.read_throughput_pb = read_throughput_pb
        self.free_load = free_load
        self.deser_wait_time = deser_wait_time
        self.deser_proc_time = deser_proc_time
        self.deser_throughput = deser_throughput
        self.deser_throughput_pb = deser_throughput_pb
        self.deser_load = deser_load
        self.comp_wait_time = comp_wait_time
        self.comp_proc_time = comp_proc_time
        self.comp_throughput = comp_throughput
        self.comp_throughput_pb = comp_throughput_pb
        self.comp_load = comp_load
        self.send_wait_time = send_wait_time
        self.send_proc_time = send_proc_time
        self.send_throughput = send_throughput
        self.send_throughput_pb = send_throughput_pb
        self.send_load = send_load

    @staticmethod
    def from_csv(file_path):
        last_line = Helpers.read_last_line(file_path)
        values = last_line.split(',')

        if len(values) != 22:
            raise ValueError("The number of values in the last line does not match the number of class attributes.")

        return MetricsServer(
            transfer_id=int(values[0]),
            total_time=float(values[1]),
            read_wait_time=float(values[2]),
            read_proc_time=float(values[3]),
            read_throughput=float(values[4]),
            read_throughput_pb=float(values[5]),
            free_load=float(values[6]),
            deser_wait_time=float(values[7]),
            deser_proc_time=float(values[8]),
            deser_throughput=float(values[9]),
            deser_throughput_pb=float(values[10]),
            deser_load=float(values[11]),
            comp_wait_time=float(values[12]),
            comp_proc_time=float(values[13]),
            comp_throughput=float(values[14]),
            comp_throughput_pb=float(values[15]),
            comp_load=float(values[16]),
            send_wait_time=float(values[17]),
            send_proc_time=float(values[18]),
            send_throughput=float(values[19]),
            send_throughput_pb=float(values[20]),
            send_load=float(values[21])
        )

    def to_dict(self):
        return {
            "transfer_id": self.transfer_id,
            "total_time": self.total_time,
            "read_wait_time": self.read_wait_time,
            "read_proc_time": self.read_proc_time,
            "read_throughput": self.read_throughput,
            "read_throughput_pb": self.read_throughput_pb,
            "read_load": self.read_load,
            "deser_wait_time": self.deser_wait_time,
            "deser_proc_time": self.deser_proc_time,
            "deser_throughput": self.deser_throughput,
            "deser_throughput_pb": self.deser_throughput_pb,
            "deser_load": self.deser_load,
            "comp_wait_time": self.comp_wait_time,
            "comp_proc_time": self.comp_proc_time,
            "comp_throughput": self.comp_throughput,
            "comp_throughput_pb": self.comp_throughput_pb,
            "comp_load": self.comp_load,
            "send_wait_time": self.send_wait_time,
            "send_proc_time": self.send_proc_time,
            "send_throughput": self.send_throughput,
            "send_throughput_pb": self.send_throughput_pb,
            "send_load": self.send_load
        }

    def get_throughput_metrics(self, toStr=True):
        dict = {
            "read_throughput": self.read_throughput,
            "read_throughput_pb": self.read_throughput_pb,
            "deser_throughput": self.deser_throughput,
            "deser_throughput_pb": self.deser_throughput_pb,
            "comp_throughput": self.comp_throughput,
            "comp_throughput_pb": self.comp_throughput_pb,
            "send_throughput": self.send_throughput,
            "send_throughput_pb": self.send_throughput_pb,
        }
        if not toStr:
            return dict
        return str(dict)

    def __str__(self):
        return str(self.to_dict())
    
    
class Helpers:

    @staticmethod
    def read_last_line(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            return last_line
