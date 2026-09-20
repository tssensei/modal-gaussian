"""Exercise the resident worker/barrier without a GPU or scene."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modal_gaussians.iteration_cache import atomic_json
from modal_gaussians.motion.neural import batch


class GPUWeightBatchTests(unittest.TestCase):
    def exercise(self, stage, paused=False, worker_failure=False, alpha=False, resumed=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("export","prepared","graph"):
                atomic_json(root/name/"manifest.json", {"defaults":{},"prepared_identity":"parent"})
            atomic_json(root/"config.json",{})
            output=root/"batch"
            if paused:
                atomic_json(output/"batch_workers.json",{"cpu_workers":2,"gpu_workers":0,"propagation_workers":4})
            completed=set()
            if resumed:
                completed.update((0, name) for name in ('prepare', 'graph', 'weights'))
            events=[]
            processes=[]
            jobs=[]
            for i in range(3):
                folder=output/f"bin_{i}"
                stages=[(name,folder/(name+".json"),[name,str(i)]) for name in ("prepare","graph","weights","train")]
                jobs.append(dict(bin=i,frequency_hz=float(i+1),folder=folder,stages=stages,
                    prepare_arguments={"output_dir": str(folder/"prepared")},
                    prepared_dir=folder/"prepared",graph_dir=folder/"graph",next=0,state="pending",pid=None,stage=None))
            owner=self

            class Process:
                def __init__(self,args,**kwargs):
                    self.pid=len(processes)+100
                    self.worker="modal_gaussians.motion.neural.control_preparation" in args
                    self.args=args
                    self.code=None
                    self.last=None
                    processes.append(self)
                    if self.worker:
                        events.append("worker-start")
                        owner.assertNotEqual(kwargs["env"].get("CUDA_VISIBLE_DEVICES"),"")
                    elif args[3]=="train":
                        owner.assertTrue(all((i,"weights") in completed for i in range(3)))
                        owner.assertIn("worker-exit",events)
                        events.append("train")

                def poll(self):
                    if self.code is not None: return self.code
                    if self.worker:
                        if worker_failure:
                            self.code=1
                            return 1
                        request=json.loads((output/"propagation_request.json").read_text())
                        if request.get("stop"):
                            self.code=0
                            events.append("worker-exit")
                            return 0
                        if request["id"]!=self.last:
                            self.last=request["id"]
                            operation = request.get('operation', 'weights')
                            key = 'output_dir' if operation == 'prepare' else 'prepared_dir'
                            index=int(Path(request["arguments"][key]).parent.name.split("_")[1])
                            if alpha and operation == 'weights':
                                owner.assertTrue(all((i,'prepare') in completed for i in range(3)))
                            if operation == 'prepare':
                                owner.assertNotIn('weight', events)
                            completed.add((index,operation))
                            atomic_json(output/"propagation_status.json",{"id":self.last,"status":"complete"})
                            events.append("weight" if operation == 'weights' else 'alpha')
                        return None
                    completed.add((int(self.args[4]),self.args[3]))
                    self.code=0
                    return 0

                def wait(self,timeout=None):
                    return self.poll()

                def terminate(self): self.code=-1

            ticks=0
            def tick(_):
                nonlocal ticks
                ticks+=1
                if paused and ticks<=3:
                    owner.assertFalse(any(p.worker for p in processes))
                if paused and ticks==3:
                    atomic_json(output/"batch_workers.json",{"cpu_workers":2,"gpu_workers":3,"propagation_workers":4})

            with patch.object(batch,"_jobs",return_value=({},jobs)), \
                 patch.object(batch,"resolve_config",return_value={"fragment":{"strategy":"component_field"}}), \
                 patch.object(batch,"training_revision",return_value="revision"), \
                 patch.object(batch,"_published",side_effect=lambda job,s,parent: (job["bin"],s[0]) in completed), \
                 patch.object(batch,"_gpu_sample"), patch.object(batch.time,"sleep",side_effect=tick), \
                 patch.object(batch.subprocess,"Popen",side_effect=Process):
                options=dict(modal_images=root/"export",prepared_dir=root/"prepared",geometry_graph_dir=root/"graph",
                    config_path=root/"config.json",output_dir=output,stage=stage,
                    cpu_workers=2,gpu_workers=3)
                if worker_failure:
                    with self.assertRaisesRegex(RuntimeError,"Batch failed"):
                        batch.run_batch(**options, alpha_backend="cupy" if alpha else "cpu")
                    self.assertNotIn("train",events)
                    return
                batch.run_batch(**options, alpha_backend="cupy" if alpha else "cpu")
            self.assertEqual(events.count("worker-start"),1)
            self.assertEqual(events.count("worker-exit"),1)
            self.assertEqual(events.count("weight"),2 if resumed else 3)
            if alpha:
                self.assertEqual(events.count('alpha'), 2 if resumed else 3)
            self.assertEqual(events.count("train"),3 if stage=="modes" else 0)
            state=json.loads((output/"batch_state.json").read_text())
            self.assertEqual(state["status"],"complete")
            self.assertIsNone(state["propagation_workers"])

    def test_modes_barrier(self): self.exercise("modes")
    def test_weights_only(self): self.exercise("weights")
    def test_zero_gpu_workers_pauses_launch(self): self.exercise("weights",paused=True)
    def test_worker_failure_never_starts_training(self): self.exercise("modes",worker_failure=True)
    def test_gpu_alpha_three_phases(self): self.exercise('modes', alpha=True)
    def test_gpu_alpha_weights_only(self): self.exercise('weights', alpha=True)
    def test_gpu_alpha_pause(self): self.exercise('modes', alpha=True, paused=True)
    def test_gpu_alpha_failure(self): self.exercise('modes', alpha=True, worker_failure=True)
    def test_gpu_alpha_resume(self): self.exercise('modes', alpha=True, resumed=True)

    def test_resident_worker_request_and_shutdown(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock
        from modal_gaussians.motion.neural import control_preparation as prep, control_propagation_gpu as gpu
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/"propagation_request.json",{"id":"one","arguments":{}})
            workspace=SimpleNamespace(stats={},close=Mock())
            def compute(**kwargs):
                self.assertEqual(kwargs["backend"],"cupy")
                self.assertIs(kwargs["workspace"],workspace)
                workspace.stats={"batches":1}
                atomic_json(root/"propagation_request.json",{"id":"stop","stop":True})
            with patch.object(gpu,"Workspace",return_value=workspace), patch.object(prep,"prepare_control_weights",side_effect=compute):
                prep.run_gpu_worker(root,os.getpid())
            record=json.loads((root/"propagation_status.json").read_text())
            self.assertEqual(record["status"],"complete")
            self.assertEqual(record["id"],"one")
            workspace.close.assert_called_once()

    def test_resident_alpha_transition_and_cleanup(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock
        from modal_gaussians.motion.neural import control_preparation as prep, control_propagation_gpu as gpu
        from modal_gaussians.motion.neural import selected_modal
        from modal_gaussians import synchronization_gpu as alpha
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                aw = SimpleNamespace(stats={}, close=Mock())
                pw = SimpleNamespace(stats={}, close=Mock())
                atomic_json(root/'propagation_request.json', {'id':'alpha','operation':'prepare','arguments':{}})
                def prepare(**kwargs):
                    self.assertIs(kwargs['alpha_workspace'], aw)
                    self.assertEqual(kwargs['alpha_backend'], 'cupy')
                    if fail:
                        raise ValueError('synthetic alpha failure')
                    atomic_json(root/'propagation_request.json', {'id':'weights','operation':'weights','arguments':{}})
                def allocate():
                    aw.close.assert_called_once()
                    return pw
                def weights(**kwargs):
                    self.assertIs(kwargs['workspace'], pw)
                    atomic_json(root/'propagation_request.json', {'id':'stop','stop':True})
                with patch.object(alpha,'Workspace',return_value=aw), patch.object(gpu,'Workspace',side_effect=allocate), \
                     patch.object(selected_modal,'prepare_selected_modal',side_effect=prepare), \
                     patch.object(prep,'prepare_control_weights',side_effect=weights):
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'synthetic alpha failure'):
                            prep.run_gpu_worker(root, os.getpid())
                    else:
                        prep.run_gpu_worker(root, os.getpid())
                record = json.loads((root/'propagation_status.json').read_text())
                self.assertEqual(record['status'], 'failed' if fail else 'complete')
                aw.close.assert_called_once()
                self.assertEqual(pw.close.call_count, 0 if fail else 1)


if __name__=="__main__": unittest.main()
