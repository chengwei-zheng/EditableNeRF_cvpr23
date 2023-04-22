#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def apply_model(rng, state, batch):
  fine_key, coarse_key = random.split(rng, 2)
  model_out = model.apply(
      {'params': state.optimizer.target['model']}, 
      batch,
      extra_params=state.extra_params,
      metadata_encoded=True,
      rngs={'fine': fine_key, 'coarse': coarse_key})
  return model_out


def loss_fn(rng, state, target, batch):
  batch['metadata'] = jax.tree_map(lambda x: x.reshape((1, -1)), 
                                   target['metadata'])
  model_out = apply_model(rng, state, batch)['fine']
  # loss = ((model_out['rgb'] - batch['rgb']) ** 2).mean(axis=-1)
  loss = jnp.abs(model_out['rgb'] - batch['rgb']).mean(axis=-1)
  return loss.mean()


def optim_step(rng, state, optimizer, batch):
  rng, key = random.split(rng, 2)
  grad_fn = jax.value_and_grad(loss_fn, argnums=2)
  loss, grad = grad_fn(key, state, optimizer.target, batch)
  grad = jax.lax.pmean(grad, axis_name='batch')
  loss = jax.lax.pmean(loss, axis_name='batch')

  optimizer = optimizer.apply_gradient(grad)

  return rng, loss, optimizer


p_optim_step = jax.pmap(optim_step, axis_name='batch')

key = random.PRNGKey(0)
key = key + jax.process_index()
keys = random.split(key, jax.local_device_count())

optimizer_def = optim.Adam(0.1)
init_metadata = evaluation.encode_metadata(
  model, 
  jax_utils.unreplicate(state.optimizer.target['model']), 
  jax.tree_map(lambda x: x[0, 0], data['metadata']))
# init_metadata = jax.tree_map(lambda x: x[0], init_metadata)
# Initialize to zero.
init_metadata = jax.tree_map(lambda x: jnp.zeros_like(x), init_metadata)
optimizer = optimizer_def.create({'metadata': init_metadata})
optimizer = jax_utils.replicate(optimizer, jax.local_devices())
devices = jax.local_devices()
batch_size = 1024


# In[ ]:


metadata_progression = []

for i in range(25):
  batch_inds = random.choice(keys[0], np.arange(train_data['rgb'].shape[0]), replace=False, shape=(batch_size,))
  batch = jax.tree_map(lambda x: x[batch_inds, ...], train_data)
  batch = datasets.prepare_data(batch)
  keys, loss, optimizer = p_optim_step(keys, state, optimizer, batch)
  loss = jax_utils.unreplicate(loss)
  metadata_progression.append(jax.tree_map(lambda x: np.array(x), jax_utils.unreplicate(optimizer.target['metadata'])))
  print(f'train_loss = {loss.item():.04f}')
  del batch


# In[ ]:


frames = []
for metadata in metadata_progression:
# metadata = jax_utils.unreplicate(optimizer.target['metadata'])
  camera = datasource.load_camera(target_id).scale(1.0)
  batch = make_batch(camera, None, metadata['encoded_warp'], metadata['encoded_hyper'])
  render = render_fn(state, batch, rng=rng)
  pred_rgb = np.array(render['rgb'])
  pred_depth_med = np.array(render['med_depth'])
  pred_depth_viz = viz.colorize(1.0 / pred_depth_med.squeeze())
  media.show_images([pred_rgb, pred_depth_viz])
  frames.append({ 
      'rgb': pred_rgb,
      'depth': pred_depth_med,
  })


# In[ ]:


media.show_image(data['rgb'])
media.show_videos([
    [d['rgb'] for d in frames],
    [viz.colorize(1/d['depth'].squeeze(), cmin=1.5, cmax=2.9) for d in frames],
], fps=10)


# In[ ]:




