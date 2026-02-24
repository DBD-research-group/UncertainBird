<template>
  <img v-if="dataUrl" :src="dataUrl" alt="QR Code" class="rounded-lg max-w-full" />
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import QRCode from 'qrcode'

const props = defineProps<{
  text: string
}>()

const dataUrl = ref<string>('')

function update() {
  if (!props.text) return
  QRCode.toDataURL(props.text, { width: 256, margin: 1 })
    .then((url) => { dataUrl.value = url })
    .catch(() => { dataUrl.value = '' })
}

watch(() => props.text, update, { immediate: true })
</script>
