local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end

  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end

  return t
end

--[[
performs language evaluation by running coco python code. 
]]
function utils.lang_eval(samples, vid_ids, score)
  local filename = 'val_sample'
  local file = io.open('coco_evaluation/'..filename .. '.txt', 'w')
  -- convert split samples to sentences
  local split_samples = ''
  local split_size = #samples
  for ix=1,split_size do
    -- format for coco code is: vid_id    sampled sentence
    sample = vid_ids[ix] .. '\t' .. utils.decode_seq(samples[ix]) .. '\n'
    split_samples = split_samples .. sample
  end
  file:write(split_samples)
  file:close()

  -- call code to evaluate the split samples
  os.execute('./misc/call_coco_eval.sh ' .. filename .. '.txt ' .. filename .. '.txt.json')

  -- retrieve scores 
  scores = utils.read_json('coco_evaluation/' .. filename ..'.json')
  return scores[score], split_samples
end

function utils.decode_seq(tensor_seq)
  seq_len = tensor_seq:size(1)
  sent = ''
  print(seq_len)
  for s=1,seq_len do
    local ix = tensor_seq:select(1, s):select(1, 1)
    if utils.ix_to_word[ix] == nil then 
      if s == 1 then 
        sent = sent .. ' a'
      end
      break
    else sent = sent .. utils.ix_to_word[ix] .. ' ' end
  end

  return sent
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

function utils.setVocab(ix_to_word)
  utils.ix_to_word = ix_to_word
end

return utils