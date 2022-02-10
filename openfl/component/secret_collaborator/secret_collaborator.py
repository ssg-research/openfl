# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Secret_Collaborator module."""

from enum import Enum
from logging import getLogger
from time import sleep
from typing import Tuple

from openfl.databases import TensorDB
from openfl.pipelines import NoCompressionPipeline
from openfl.pipelines import TensorCodec
from openfl.protocols import utils
from openfl.utilities import TensorKey
from openfl.component import Collaborator

import torchvision
import torch
import glob
import numpy as np
import warnings

from PIL import Image as Img
# from matplotlib import pyplot as plt
#import matplotlib
import param
from numpy import pi
from param.parameterized import ParamOverrides
from contextlib import contextmanager
from holoviews import HoloMap, Image, Dimension
from holoviews.core import BoundingBox, BoundingRegionParameter, SheetCoordinateSystem

from openfl.federated.task import FederatedModel
from openfl.federated.data import FederatedDataSet
import torch.optim as optim

class DevicePolicy(Enum):
    """Device assignment policy."""

    CPU_ONLY = 1

    CUDA_PREFERRED = 2


class OptTreatment(Enum):
    """Optimizer Methods."""

    RESET = 1
    '''
    RESET tells each collaborator to reset the optimizer state at the beginning
    of each round.
    '''
    CONTINUE_LOCAL = 2
    '''
    CONTINUE_LOCAL tells each collaborator to continue with the local optimizer
    state from the previous round.
    '''
    CONTINUE_GLOBAL = 3
    '''
    CONTINUE_GLOBAL tells each collaborator to continue with the federally
    averaged optimizer state from the previous round.
    '''

class Pattern(torch.utils.data.Dataset):
    # This is used to load images from the defined pattern image file

    def __init__(self, root_dir: str, n_classes: int, transform: torchvision.transforms.Compose,
                 return_image_path: bool = False) -> None:
        """
        Args:
            root_dir(string): file containing name of all images
            train (bool): gets only train or test set (not used here since ImageNet is used for out-of-distribution data only, used for API consistency)
            transform (torch func): transforms PIL image to torch data
            download (bool): indicates if the ImageNet should be downloaded (used for API consistency, doesn't do anything here)
            nrows (int, opt): gets only first nrows image to the dataset
            use_probs(bool, optional): labels will be avector containing probabilities for each class
        """

        import torchvision.transforms as transforms
        #from PIL import Image
        import os

        self.root_dir = root_dir
        self.transform = transform
        self.n_classes = n_classes

        self.train_data = []
        self.train_labels = []
        for i in range(0, self.n_classes):
            dirs = os.path.join(self.root_dir, '%d' % i)
            for idx, fimg in enumerate(glob.glob(os.path.join(dirs, '*.png'))):
                if idx < 10:
                    image = Img.open(fimg)
                    #image = image.convert('RGB')
                    image = self.transform(image)
                    #image = np.array(self.transform(image.convert('RGB')))
                    # below is true just for mnist
                    image = (image*255)*0.5 + (255*0.5)
                    self.train_data.append(image)
                    self.train_labels.append(i)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        image = self.data[idx]
        #image = self.transform(image)
        label = self.labels[idx]
        return image, label

    #@property
    #def train_data(self):
    #    warnings.warn("train_data has been renamed data")
    #    return torch.tensor(self.data)

    #@property
    #def train_labels(self):
    #    warnings.warn("train_labels has been renamed targets")
    #    return self.labels


class Secret_Collaborator(Collaborator):
    r"""The Collaborator object class.

    Args:
        collaborator_name (string): The common name for the collaborator
        aggregator_uuid: The unique id for the client
        federation_uuid: The unique id for the federation
        model: The model
        opt_treatment* (string): The optimizer state treatment (Defaults to
            "CONTINUE_GLOBAL", which is aggreagated state from previous round.)

        compression_pipeline: The compression pipeline (Defaults to None)

        num_batches_per_round (int): Number of batches per round
                                     (Defaults to None)

        delta_updates* (bool): True = Only model delta gets sent.
                               False = Whole model gets sent to collaborator.
                               Defaults to False.

        single_col_cert_common_name: (Defaults to None)

    Note:
        \* - Plan setting.
    """

    def __init__(self,
                 collaborator_name,
                 aggregator_uuid,
                 federation_uuid,
                 client,
                 task_runner,
                 task_config,
                 opt_treatment='RESET',
                 device_assignment_policy='CPU_ONLY',
                 delta_updates=False,
                 compression_pipeline=None,
                 db_store_rounds=1,
                 watermark_number=10,
                 learning_rate=0.001,
                 watermark_batch_size=50,
                 **kwargs):
        """Initialize."""
        self.single_col_cert_common_name = None

        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = ''  # for protobuf compatibility
        # we would really want this as an object

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.tensor_db = TensorDB()
        self.db_store_rounds = db_store_rounds

        self.task_runner = task_runner
        self.delta_updates = delta_updates

        self.client = client

        self.task_config = task_config

        self.logger = getLogger(__name__)

        self.watermark_class = task_runner.data_loader.num_classes
        self.watermark_number = watermark_number
        self.learning_rate = learning_rate
        self.watermark_batch_size = watermark_batch_size

        self.set_task_runner(task_runner)

        # RESET/CONTINUE_LOCAL/CONTINUE_GLOBAL
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error(f'Unknown opt_treatment: {opt_treatment.name}.')
            raise NotImplementedError(f'Unknown opt_treatment: {opt_treatment}.')

        if hasattr(DevicePolicy, device_assignment_policy):
            self.device_assignment_policy = DevicePolicy[device_assignment_policy]
        else:
            self.logger.error('Unknown device_assignment_policy: '
                              f'{device_assignment_policy.name}.')
            raise NotImplementedError(
                f'Unknown device_assignment_policy: {device_assignment_policy}.'
            )

        self.task_runner.set_optimizer_treatment(self.opt_treatment.name)

    def set_task_runner(self, runner):

        x_input = runner.feature_shape[-2]
        y_input = runner.feature_shape[-1]

        wm_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            #torchvision.transforms.Resize(x_input),
            #torchvision.transforms.CenterCrop(x_input),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])
        watermark_data_path = './data/WATERMARK/'
        self.generate_mpattern(x_input=x_input, y_input=y_input, num_class=self.watermark_class, num_picures=self.watermark_number, watermark_data_path=watermark_data_path)
        watermark_set = Pattern(watermark_data_path, n_classes=self.watermark_class, transform=wm_transform)
        watermark_images, watermark_labels = torch.stack(watermark_set.train_data), np.array(watermark_set.train_labels)
        y_valid_watermark = torch.nn.functional.one_hot(torch.tensor(watermark_labels)).numpy()
        watermark_data = FederatedDataSet(watermark_images, watermark_labels, watermark_images, y_valid_watermark,
                                          batch_size=self.watermark_batch_size,
                                          num_classes=self.watermark_class)

        #optimizer_watermark = lambda x: optim.Adam(x, lr=self.learning_rate)
        optimizer_watermark = lambda x: optim.SGD(x, lr=self.learning_rate, momentum=0.5, weight_decay=0.00005)

        watermark_model = FederatedModel(build_model=runner.build_model,
                                         optimizer=optimizer_watermark, loss_fn=self.cross_entropy,
                                         data_loader=watermark_data)
        watermark_collaborator_models = watermark_model.setup(num_collaborators=1)[0]

        self.task_runner = watermark_collaborator_models

    def cross_entropy(self, output, target):
        """Binary cross-entropy metric
        """
        import torch.nn.functional as F
        return F.cross_entropy(input=output, target=target)

    def generate_mpattern(self, x_input, y_input, num_class, num_picures, watermark_data_path):
        #import matplotlib
        #import matplotlib.pyplot as plt
        import numbergen as ng
        import os

        x_pattern = int(x_input * 2 / 3. - 1)
        y_pattern = int(y_input * 2 / 3. - 1)

        for cls in range(num_class):
            # define patterns
            patterns = []
            patterns.append(
                Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                        x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
            patterns.append(
                Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                       x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

            pat = np.zeros((x_pattern, y_pattern))
            for i in range(6):
                j = np.random.randint(len(patterns))
                pat += patterns[j]()
            res = pat > 0.5
            pat = res.astype(int)
            # print(pat)

            x_offset = np.random.randint(x_input - x_pattern + 1)
            y_offset = np.random.randint(y_input - y_pattern + 1)
            # print(x_offset, y_offset)

            for i in range(num_picures):
                base = np.random.rand(x_input, y_input)
                base[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] += pat
                d = np.ones((x_input, x_input))
                img = np.minimum(base, d)
                if not os.path.exists(watermark_data_path + str(cls) + "/"):
                    os.makedirs(watermark_data_path + str(cls) + "/")
                image = Img.fromarray((img*255).astype(np.uint8))
                image.save(watermark_data_path + str(cls) + "/wm_" + str(i + 1) + ".png")

    def set_available_devices(self, cuda: Tuple[str] = ()):
        """
        Set available CUDA devices.

        Cuda tuple contains string indeces, ('1', '3').
        """
        self.cuda_devices = cuda

    def run(self):
        """Run the collaborator."""
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                break
            elif sleep_time > 0:
                sleep(sleep_time)  # some sleep function
            else:
                self.logger.info(f'Received the following tasks: {tasks}')
                for task in tasks:
                    self.do_task(task, round_number)

                # Cleaning tensor db
                self.tensor_db.clean_up(self.db_store_rounds)

        self.logger.info('End of Federation reached. Exiting...')

    def run_simulation(self):
        """
        Specific function for the simulation.

        After the tasks have
        been performed for a roundquit, and then the collaborator object will
        be reinitialized after the next round
        """
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            # round_number -= 1
            if time_to_quit:
                self.logger.info('End of Federation reached. Exiting...')
                break
            elif sleep_time > 0:
                sleep(sleep_time)  # some sleep function
            else:
                self.logger.info(f'Received the following tasks: {tasks}')
                for task in tasks:
                    self.do_task(task, round_number)
                self.logger.info(f'All tasks completed on {self.collaborator_name} '
                                 f'for round {round_number}...')
                break

    def get_tasks(self):
        """Get tasks from the aggregator."""
        # logging wait time to analyze training process
        self.logger.info('Waiting for tasks...')

##########################################################
        if self.collaborator_name=='secret':
            tasks, round_number, sleep_time, time_to_quit = self.client.get_tasks(
                self.client.authorized_cols[0])
        else:
            tasks, round_number, sleep_time, time_to_quit = self.client.get_tasks(
                self.collaborator_name)
##########################################################
        return tasks, round_number, sleep_time, time_to_quit

    def do_task(self, task, round_number):
        """Do the specified task."""
        # map this task to an actual function name and kwargs
        func_name = self.task_config[task]['function']
        kwargs = self.task_config[task]['kwargs']

        # this would return a list of what tensors we require as TensorKeys
        required_tensorkeys_relative = self.task_runner.get_required_tensorkeys_for_function(
            func_name,
            **kwargs
        )

        # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL,
        # round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = []
        for tname, origin, rnd_num, report, tags in required_tensorkeys_relative:
            if origin == 'GLOBAL':
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name

            # rnd_num is the relative round. So if rnd_num is -1, get the
            # tensor from the previous round
            required_tensorkeys.append(
                TensorKey(tname, origin, rnd_num + round_number, report, tags)
            )

        # print('Required tensorkeys = {}'.format(
        # [tk[0] for tk in required_tensorkeys]))
        input_tensor_dict = self.get_numpy_dict_for_tensorkeys(
            required_tensorkeys
        )

        # now we have whatever the model needs to do the task
        if hasattr(self.task_runner, 'TASK_REGISTRY'):
            # New interactive python API
            # New `Core` TaskRunner contains registry of tasks
            func = self.task_runner.TASK_REGISTRY[func_name]
            self.logger.info('Using Interactive Python API')

            # So far 'kwargs' contained parameters read from the plan
            # those are parameters that the eperiment owner registered for
            # the task.
            # There is another set of parameters that created on the
            # collaborator side, for instance, local processing unit identifier:s
            if (self.device_assignment_policy is DevicePolicy.CUDA_PREFERRED
                    and len(self.cuda_devices) > 0):
                kwargs['device'] = f'cuda:{self.cuda_devices[0]}'
            else:
                kwargs['device'] = 'cpu'
        else:
            # TaskRunner subclassing API
            # Tasks are defined as methods of TaskRunner
            func = getattr(self.task_runner, func_name)
            self.logger.info('Using TaskRunner subclassing API')
        if 'train' in func_name:
            global_output_tensor_dict, local_output_tensor_dict = func(
                col_name=self.collaborator_name,
                round_num=round_number-1,
                input_tensor_dict=input_tensor_dict,
                #epochs=100,
                **kwargs)
        else:
            global_output_tensor_dict, local_output_tensor_dict = func(
                col_name=self.collaborator_name,
                round_num=round_number - 1,
                input_tensor_dict=input_tensor_dict,
                **kwargs)


        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        self.send_task_results(global_output_tensor_dict, round_number, task)

    def get_numpy_dict_for_tensorkeys(self, tensor_keys):
        """Get tensor dictionary for specified tensorkey set."""
        return {k.tensor_name: self.get_data_for_tensorkey(k) for k in tensor_keys}

    def get_data_for_tensorkey(self, tensor_key):
        """
        Resolve the tensor corresponding to the requested tensorkey.

        Args
        ----
        tensor_key:         Tensorkey that will be resolved locally or
                            remotely. May be the product of other tensors
        """
        # try to get from the store
        tensor_name, origin, round_number, report, tags = tensor_key
        self.logger.debug(f'Attempting to retrieve tensor {tensor_key} from local store')
        nparray = self.tensor_db.get_tensor_from_cache(tensor_key)

        # if None and origin is our client, request it from the client
        if nparray is None:
            if origin == self.collaborator_name:
                self.logger.info(
                    f'Attempting to find locally stored {tensor_name} tensor from prior round...'
                )
                prior_round = round_number-1
                while prior_round >= 0:
                    nparray = self.tensor_db.get_tensor_from_cache(
                        TensorKey(tensor_name, origin, prior_round, report, tags))
                    if nparray is not None:
                        self.logger.debug(f'Found tensor {tensor_name} in local TensorDB '
                                          f'for round {prior_round}')
                        return nparray
                    prior_round -= 1
                self.logger.info(
                    f'Cannot find any prior version of tensor {tensor_name} locally...'
                )
            self.logger.debug('Unable to get tensor from local store...'
                              'attempting to retrieve from client')
            # Determine whether there are additional compression related
            # dependencies.
            # Typically, dependencies are only relevant to model layers
            tensor_dependencies = self.tensor_codec.find_dependencies(
                tensor_key, self.delta_updates
            )
            if len(tensor_dependencies) > 0:
                # Resolve dependencies
                # tensor_dependencies[0] corresponds to the prior version
                # of the model.
                # If it exists locally, should pull the remote delta because
                # this is the least costly path
                prior_model_layer = self.tensor_db.get_tensor_from_cache(
                    tensor_dependencies[0]
                )
                if prior_model_layer is not None:
                    uncompressed_delta = self.get_aggregated_tensor_from_aggregator(
                        tensor_dependencies[1]
                    )
                    new_model_tk, nparray = self.tensor_codec.apply_delta(
                        tensor_dependencies[1],
                        uncompressed_delta,
                        prior_model_layer,
                        creates_model=True,
                    )
                    self.tensor_db.cache_tensor({new_model_tk: nparray})
                else:
                    self.logger.info('Count not find previous model layer.'
                                     'Fetching latest layer from aggregator')
                    # The original model tensor should be fetched from client
                    nparray = self.get_aggregated_tensor_from_aggregator(
                        tensor_key,
                        require_lossless=True
                    )
            elif 'model' in tags:
                # Pulling the model for the first time
                nparray = self.get_aggregated_tensor_from_aggregator(
                    tensor_key,
                    require_lossless=True
                )
        else:
            self.logger.debug(f'Found tensor {tensor_key} in local TensorDB')

        return nparray

    def get_aggregated_tensor_from_aggregator(self, tensor_key,
                                              require_lossless=False):
        """
        Return the decompressed tensor associated with the requested tensor key.

        If the key requests a compressed tensor (in the tag), the tensor will
        be decompressed before returning
        If the key specifies an uncompressed tensor (or just omits a compressed
        tag), the decompression operation will be skipped

        Args
        ----
        tensor_key  :               The requested tensor
        require_lossless:   Should compression of the tensor be allowed
                                    in flight?
                                    For the initial model, it may affect
                                    convergence to apply lossy
                                    compression. And metrics shouldn't be
                                    compressed either

        Returns
        -------
        nparray     : The decompressed tensor associated with the requested
                      tensor key
        """
        tensor_name, origin, round_number, report, tags = tensor_key

        self.logger.debug(f'Requesting aggregated tensor {tensor_key}')
        tensor = self.client.get_aggregated_tensor(
            self.collaborator_name, tensor_name, round_number, report, tags, require_lossless)

        # this translates to a numpy array and includes decompression, as
        # necessary
        nparray = self.named_tensor_to_nparray(tensor)

        # cache this tensor
        self.tensor_db.cache_tensor({tensor_key: nparray})

        return nparray

    def send_task_results(self, tensor_dict, round_number, task_name):
        """Send task results to the aggregator."""
        round_number-=1
        named_tensors = [
            self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()
        ]

        # for general tasks, there may be no notion of data size to send.
        # But that raises the question how to properly aggregate results.

        data_size = -1

        if 'train' in task_name:
            data_size = self.task_runner.get_train_data_size()

        if 'valid' in task_name:
            data_size = self.task_runner.get_valid_data_size()

        self.logger.debug(f'{task_name} data size = {data_size}')

        for tensor in tensor_dict:
            tensor_name, origin, fl_round, report, tags = tensor

            if report:
                self.logger.metric(
                    f'Round {round_number}, collaborator {self.collaborator_name} '
                    f'is sending metric for task {task_name}:'
                    f' {tensor_name}\t{tensor_dict[tensor]}')

        self.client.send_watermark_results(
            self.collaborator_name, round_number, task_name, data_size, named_tensors)

    def nparray_to_named_tensor(self, tensor_key, nparray):
        """
        Construct the NamedTensor Protobuf.

        Includes logic to create delta, compress tensors with the TensorCodec, etc.
        """
        # if we have an aggregated tensor, we can make a delta
        tensor_name, origin, round_number, report, tags = tensor_key
        if 'trained' in tags and self.delta_updates:
            # Should get the pretrained model to create the delta. If training
            # has happened,
            # Model should already be stored in the TensorDB
            model_nparray = self.tensor_db.get_tensor_from_cache(
                TensorKey(
                    tensor_name,
                    origin,
                    round_number,
                    report,
                    ('model',)
                )
            )

            # The original model will not be present for the optimizer on the
            # first round.
            if model_nparray is not None:
                delta_tensor_key, delta_nparray = self.tensor_codec.generate_delta(
                    tensor_key,
                    nparray,
                    model_nparray
                )
                delta_comp_tensor_key, delta_comp_nparray, metadata = self.tensor_codec.compress(
                    delta_tensor_key,
                    delta_nparray
                )

                named_tensor = utils.construct_named_tensor(
                    delta_comp_tensor_key,
                    delta_comp_nparray,
                    metadata,
                    lossless=False
                )
                return named_tensor

        # Assume every other tensor requires lossless compression
        compressed_tensor_key, compressed_nparray, metadata = self.tensor_codec.compress(
            tensor_key,
            nparray,
            require_lossless=True
        )
        named_tensor = utils.construct_named_tensor(
            compressed_tensor_key,
            compressed_nparray,
            metadata,
            lossless=True
        )

        return named_tensor

    def named_tensor_to_nparray(self, named_tensor):
        """Convert named tensor to a numpy array."""
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [{'int_to_float': proto.int_to_float,
                     'int_list': proto.int_list,
                     'bool_list': proto.bool_list
                     } for proto in named_tensor.transformer_metadata]
        # The tensor has already been transfered to collaborator, so
        # the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self.collaborator_name,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags)
        )
        tensor_name, origin, round_number, report, tags = tensor_key
        if 'compressed' in tags:
            decompressed_tensor_key, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=True
            )
        elif 'lossy_compressed' in tags:
            decompressed_tensor_key, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata
            )
        else:
            # There could be a case where the compression pipeline is bypassed
            # entirely
            self.logger.warning('Bypassing tensor codec...')
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        self.tensor_db.cache_tensor(
            {decompressed_tensor_key: decompressed_nparray}
        )

        return decompressed_nparray

@contextmanager
def float_error_ignore():
    oldsettings=np.seterr(divide='ignore', under='ignore')
    yield
    np.seterr(**oldsettings)


class PatternGenerator(param.Parameterized):
    __abstract = True

    bounds = BoundingRegionParameter(
        default=BoundingBox(points=((-0.5, -0.5), (0.5, 0.5))), precedence=-1)

    xdensity = param.Number(default=256, bounds=(0, None), precedence=-1)
    ydensity = param.Number(default=256, bounds=(0, None), precedence=-1)

    x = param.Number(default=0.0, softbounds=(-1.0, 1.0), precedence=0.20)
    y = param.Number(default=0.0, softbounds=(-1.0, 1.0), precedence=0.21)
    z = param.ClassSelector(default=None, precedence=-1, class_=Dimension)

    group = param.String(default='Pattern', precedence=-1)

    position = param.Composite(attribs=['x', 'y'], precedence=-1)
    orientation = param.Number(default=0.0, softbounds=(0.0, 2 * pi), precedence=0.40)

    size = param.Number(default=1.0, bounds=(0.0, None), softbounds=(0.0, 6.0),
                        precedence=0.30)

    scale = param.Number(default=1.0, softbounds=(0.0, 2.0), precedence=0.10)

    offset = param.Number(default=0.0, softbounds=(-1.0, 1.0), precedence=0.11)

    mask = param.Parameter(default=None, precedence=-1)

    mask_shape = param.ClassSelector(param.Parameterized, default=None, precedence=0.06)

    output_fns = param.HookList(default=[], precedence=0.08)

    def __init__(self, **params):
        super(PatternGenerator, self).__init__(**params)
        self.set_matrix_dimensions(self.bounds, self.xdensity, self.ydensity)

    def __call__(self, **params_to_override):
        p = ParamOverrides(self, params_to_override)

        self._setup_xy(p.bounds, p.xdensity, p.ydensity, p.x, p.y, p.orientation)
        fn_result = self.function(p)
        self._apply_mask(p, fn_result)
        if p.scale != 1.0:
            result = p.scale * fn_result
        else:
            result = fn_result
        if p.offset != 0.0:
            result += p.offset

        for of in p.output_fns:
            of(result)

        return result

    def __getitem__(self, coords):
        value_dims = {}

        raster, data = Image, self()
        value_dims = {'value_dimensions': [self.z]} if self.z else value_dims

        image = raster(data, bounds=self.bounds,
                       **dict(group=self.group,
                              label=self.__class__.__name__, **value_dims))
        # Works round a bug fixed shortly after HoloViews 1.0.0 release
        return image if isinstance(coords, slice) else image.__getitem__(coords)

    def _setup_xy(self, bounds, xdensity, ydensity, x, y, orientation):
        x_points, y_points = SheetCoordinateSystem(bounds, xdensity, ydensity).sheetcoordinates_of_matrixidx()

        self.pattern_x, self.pattern_y = self._create_and_rotate_coordinate_arrays(x_points - x, y_points - y,
                                                                                   orientation)

    def _create_and_rotate_coordinate_arrays(self, x, y, orientation):

        pattern_y = np.subtract.outer(np.cos(orientation) * y, np.sin(orientation) * x)
        pattern_x = np.add.outer(np.sin(orientation) * y, np.cos(orientation) * x)
        return pattern_x, pattern_y

    def _apply_mask(self, p, mat):
        mask = p.mask
        ms = p.mask_shape
        if ms is not None:
            mask = ms(x=p.x + p.size * (ms.x * np.cos(p.orientation) - ms.y * np.sin(p.orientation)),
                      y=p.y + p.size * (ms.x * np.sin(p.orientation) + ms.y * np.cos(p.orientation)),
                      orientation=ms.orientation + p.orientation, size=ms.size * p.size,
                      bounds=p.bounds, ydensity=p.ydensity, xdensity=p.xdensity)
        if mask is not None:
            mat *= mask

    def set_matrix_dimensions(self, bounds, xdensity, ydensity):
        self.bounds = bounds
        self.xdensity = xdensity
        self.ydensity = ydensity


class Line(PatternGenerator):
    size = param.Number(precedence=-1.0)

    thickness = param.Number(default=0.006, bounds=(0.0, None), softbounds=(0.0, 1.0),
                             precedence=0.60)
    enforce_minimal_thickness = param.Boolean(default=False, precedence=0.60)

    smoothing = param.Number(default=0.05, bounds=(0.0, None), softbounds=(0.0, 0.5),
                             precedence=0.61)

    def function(self, p):
        distance_from_line = abs(self.pattern_y)
        gaussian_y_coord = distance_from_line - p.thickness / 2.0
        sigmasq = p.smoothing * p.smoothing

        if sigmasq == 0.0:
            falloff = self.pattern_y * 0.0
        else:
            with float_error_ignore():
                falloff = np.exp(np.divide(-gaussian_y_coord * gaussian_y_coord, 2 * sigmasq))

        return np.where(gaussian_y_coord <= 0, 1.0, falloff)

class Arc(PatternGenerator):
    aspect_ratio = param.Number(default=1.0, bounds=(0.0, None), softbounds=(0.0, 6.0), precedence=0.31)

    thickness = param.Number(default=0.015, bounds=(0.0, None), softbounds=(0.0, 0.5), precedence=0.60)

    smoothing = param.Number(default=0.05, bounds=(0.0, None), softbounds=(0.0, 0.5), precedence=0.61)

    arc_length = param.Number(default=pi, bounds=(0.0, None), softbounds=(0.0, 2.0*pi),
                              inclusive_bounds=(True, False), precedence=0.62)

    size = param.Number(default=0.5)

    def arc_by_radian(self, x, y, height, radian_range, thickness, gaussian_width):
        radius = height / 2.0
        half_thickness = thickness / 2.0

        distance_from_origin = np.sqrt(x ** 2 + y ** 2)
        distance_outside_outer_disk = distance_from_origin - radius - half_thickness
        distance_inside_inner_disk = radius - half_thickness - distance_from_origin

        ring = 1.0 - np.bitwise_xor(np.greater_equal(distance_inside_inner_disk, 0.0),
                                    np.greater_equal(distance_outside_outer_disk, 0.0))

        sigmasq = gaussian_width * gaussian_width

        if sigmasq == 0.0:
            inner_falloff = x * 0.0
            outer_falloff = x * 0.0
        else:
            with float_error_ignore():
                inner_falloff = np.exp(
                    np.divide(-distance_inside_inner_disk * distance_inside_inner_disk, 2.0 * sigmasq))
                outer_falloff = np.exp(
                    np.divide(-distance_outside_outer_disk * distance_outside_outer_disk, 2.0 * sigmasq))

        output_ring = np.maximum(inner_falloff, np.maximum(outer_falloff, ring))

        distance_from_origin += np.where(distance_from_origin == 0.0, 1e-5, 0)

        with float_error_ignore():
            sines = np.divide(y, distance_from_origin)
            cosines = np.divide(x, distance_from_origin)
            arcsines = np.arcsin(sines)

        phase_1 = np.where(np.logical_and(sines >= 0, cosines >= 0), 2 * pi - arcsines, 0)
        phase_2 = np.where(np.logical_and(sines >= 0, cosines < 0), pi + arcsines, 0)
        phase_3 = np.where(np.logical_and(sines < 0, cosines < 0), pi + arcsines, 0)
        phase_4 = np.where(np.logical_and(sines < 0, cosines >= 0), -arcsines, 0)
        arcsines = phase_1 + phase_2 + phase_3 + phase_4

        if radian_range[0] <= radian_range[1]:
            return np.where(np.logical_and(arcsines >= radian_range[0], arcsines <= radian_range[1]),
                            output_ring, 0.0)
        else:
            return np.where(np.logical_or(arcsines >= radian_range[0], arcsines <= radian_range[1]),
                            output_ring, 0.0)

    def function(self, p):
        if p.aspect_ratio == 0.0:
            return self.pattern_x*0.0

        return self.arc_by_radian(self.pattern_x/p.aspect_ratio, self.pattern_y, p.size,
                             (2*pi-p.arc_length, 0.0), p.thickness, p.smoothing)
